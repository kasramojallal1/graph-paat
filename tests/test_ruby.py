"""Ruby, read through tree-sitter.

The point of another language is not that it works, but that it produces the
SAME shapes: a Ruby class is a `class` node, so is a module, `include` is
`inherits`, and a run of `#` lines is a `rationale`. If any of that needed a new
word, the model was never language-neutral and everything downstream would have
to learn Ruby too.

Ruby's own awkwardness is tested here as well. It nests everything in a module
and reopens classes across files, so what a name is scoped by has to be pinned
down; `require` is an ordinary method call rather than a statement; and `#`
carries both the documentation and every linter pragma in the file.
"""
import pytest

from graphpaat.parse import parse_corpus_files, parse_file
from graphpaat.resolve import resolve

pytest.importorskip("tree_sitter_ruby")


@pytest.fixture
def rbcorpus(tmp_path):
    def build(files: dict[str, str]):
        root = tmp_path / "repo"
        for name, source in files.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(source, encoding="utf-8")
        root.mkdir(parents=True, exist_ok=True)
        return root
    return build


def kinds(root):
    files, _ = parse_corpus_files(root)
    return {n.id: n.kind for p in files for n in p.nodes}


def links(root, relation="calls"):
    """Resolved edges of one relation.

    Containment comes from parsing and calls come from resolution, so both
    sources are needed -- looking only at the resolver's output made a correct
    containment edge look missing.
    """
    files, _ = parse_corpus_files(root)
    edges, _ = resolve(files)
    edges = list(edges) + [e for p in files for e in p.edges]
    return {(e.source, e.target) for e in edges
            if e.relation == relation and e.resolved}


def reasons_for(root):
    files, _ = parse_corpus_files(root)
    return resolve(files)[1]


def docs(root):
    files, _ = parse_corpus_files(root)
    return {n.id: n.text for p in files for n in p.nodes if n.kind == "rationale"}


SERVER = {"srv.rb": '''# frozen_string_literal: true

module Rack
  # Server accepts connections.
  class Server
    # Start begins serving.
    def start
      listen
      self.finish
    end

    def listen; end

    def finish; end
  end
end
'''}


class TestSameShapesAsPython:
    def test_a_class_is_a_class_node(self, rbcorpus):
        assert kinds(rbcorpus(SERVER))["srv_server"] == "class"

    def test_a_module_is_also_a_class_node(self, rbcorpus):
        # A module is a namespace and, once included, part of another type's
        # ancestry. Both are what a `class` node means here.
        assert kinds(rbcorpus(SERVER))["srv_rack"] == "class"

    def test_methods_are_qualified_by_their_class(self, rbcorpus):
        k = kinds(rbcorpus(SERVER))
        assert k["srv_server_start"] == "method"

    def test_a_def_outside_any_class_is_a_function(self, rbcorpus):
        root = rbcorpus({"m.rb": "def helper(a)\n  a\nend\n"})
        assert kinds(root)["m_helper"] == "function"

    def test_a_singleton_method_is_a_method(self, rbcorpus):
        root = rbcorpus({"m.rb": "class Builder\n  def self.parse(x); end\nend\n"})
        assert kinds(root)["m_builder_parse"] == "method"

    def test_a_def_inside_class_shift_self_belongs_to_the_class(self, rbcorpus):
        # `class << self` opens the singleton of the class around it. Emitting a
        # node for it would put every class method under a nameless owner.
        root = rbcorpus({"m.rb": '''class Builder
  class << self
    def app; end
  end
end
'''})
        k = kinds(root)
        assert k["m_builder_app"] == "method"
        assert ("m_builder", "m_builder_app") in links(root, "contains")

    def test_two_classes_with_a_same_named_method_do_not_collide(self, rbcorpus):
        root = rbcorpus({"m.rb": '''class A
  def run; end
end
class B
  def run; end
end
'''})
        k = kinds(root)
        assert "m_a_run" in k and "m_b_run" in k

    def test_containment_matches_the_python_shape(self, rbcorpus):
        got = links(rbcorpus(SERVER), "contains")
        assert ("srv", "srv_rack") in got             # file contains the module
        assert ("srv_rack", "srv_server") in got      # module contains the class
        assert ("srv_server", "srv_server_start") in got

    def test_a_nested_class_is_minted_flat_but_contained(self, rbcorpus):
        # The enclosing module stays out of the id: a call site asking for
        # `Headers` rebuilds `<file>_headers`, and putting `Request` in the name
        # would mean nothing ever found it. The nesting survives as an edge.
        root = rbcorpus({"m.rb": '''class Request
  class Headers
    def add; end
  end
end
'''})
        k = kinds(root)
        assert k["m_headers"] == "class"
        assert k["m_headers_add"] == "method"
        assert ("m_request", "m_headers") in links(root, "contains")


class TestDocComments:
    def test_a_comment_run_becomes_a_rationale_node(self, rbcorpus):
        got = docs(rbcorpus(SERVER))
        assert "Server accepts connections." in got["srv_server#doc"]
        assert "Start begins serving." in got["srv_server_start#doc"]

    def test_the_first_class_in_a_module_still_finds_its_comment(self, rbcorpus):
        # The grammar hangs the comments that open a module body off the module
        # node, ahead of the body. Looking only at siblings loses the
        # documentation of the first class in nearly every Ruby file.
        assert "srv_server#doc" in docs(rbcorpus(SERVER))

    def test_a_comment_separated_by_a_blank_line_is_not_documentation(self, rbcorpus):
        root = rbcorpus({"m.rb": '''# Unrelated note.

class Thing; end
'''})
        assert docs(root) == {}

    def test_a_trailing_comment_does_not_document_the_next_definition(self, rbcorpus):
        # `X = 1  # note` sits one row above the next def and passed the
        # "immediately above" test, making every trailing note a docstring.
        root = rbcorpus({"m.rb": '''class Thing
  X = 1  # a note about X
  def run; end
end
'''})
        assert "m_thing_run#doc" not in docs(root)

    def test_a_magic_comment_is_not_documentation(self, rbcorpus):
        root = rbcorpus({"m.rb": "# frozen_string_literal: true\nclass Thing; end\n"})
        assert docs(root) == {}

    def test_a_linter_pragma_is_not_documentation(self, rbcorpus):
        root = rbcorpus({"m.rb": '''class Thing
  # rubocop:disable Metrics/AbcSize
  def run; end
end
'''})
        assert "m_thing_run#doc" not in docs(root)

    def test_a_word_that_looks_like_a_magic_comment_is_documentation(self, rbcorpus):
        # The colon is what separates a pragma from a sentence. Without it,
        # anything whose first word was "Encoding" lost its documentation.
        root = rbcorpus({"m.rb": '''class Thing
  # Encoding of the response body.
  def run; end
end
'''})
        assert "Encoding of the response body." in docs(root)["m_thing_run#doc"]

    def test_a_begin_end_block_is_documentation(self, rbcorpus):
        root = rbcorpus({"m.rb": '''=begin
Thing does the thing.
=end
class Thing; end
'''})
        assert "Thing does the thing." in docs(root)["m_thing#doc"]


class TestInheritance:
    def test_a_superclass_is_inheritance(self, rbcorpus):
        root = rbcorpus({"m.rb": '''class Base; end
class Derived < Base; end
'''})
        assert ("m_derived", "m_base") in links(root, "inherits")

    def test_include_is_inheritance(self, rbcorpus):
        # `include` puts the module in the ancestor chain and its methods
        # become the class's methods -- the same thing embedding means in Go
        # and `implements` in TypeScript, both already drawn as `inherits`.
        root = rbcorpus({"m.rb": '''module Helpers; end
class Request
  include Helpers
end
'''})
        assert ("m_request", "m_helpers") in links(root, "inherits")

    def test_a_qualified_superclass_resolves_by_its_last_name(self, rbcorpus):
        root = rbcorpus({"a.rb": "module Rack\n  class Base; end\nend\n",
                         "b.rb": "class Derived < Rack::Base; end\n"})
        assert ("b_derived", "a_base") in links(root, "inherits")

    def test_a_base_outside_the_corpus_is_not_invented(self, rbcorpus):
        root = rbcorpus({"m.rb": "class Boom < StandardError; end\n"})
        assert reasons_for(root)["base class outside this corpus"] >= 1


class TestNamesAndReopening:
    def test_class_written_with_a_scope_is_named_by_its_last_segment(self, rbcorpus):
        # `class Rack::Request` and `module Rack; class Request` are the same
        # class written two ways, so they have to arrive under one name.
        root = rbcorpus({"m.rb": "class Rack::Request\n  def go; end\nend\n"})
        k = kinds(root)
        assert k["m_request"] == "class"
        assert k["m_request_go"] == "method"

    def test_a_reopened_module_is_one_node(self, rbcorpus):
        root = rbcorpus({"m.rb": '''module Rack; end

module Rack
  class Builder; end
end
'''})
        files, _ = parse_corpus_files(root)
        ids = [n.id for p in files for n in p.nodes]
        assert ids.count("m_rack") == 1

    def test_two_different_classes_of_one_name_are_still_two_nodes(self, rbcorpus):
        # The flat id scheme cannot separate `A::Thing` from `B::Thing` in one
        # file. They are emitted anyway, so the collision report names them,
        # rather than being silently folded together.
        root = rbcorpus({"m.rb": '''module A
  class Thing; end
end
module B
  class Thing; end
end
'''})
        files, _ = parse_corpus_files(root)
        ids = [n.id for p in files for n in p.nodes]
        assert ids.count("m_thing") == 2

    def test_a_def_hidden_in_a_conditional_is_still_found(self, rbcorpus):
        # A class body that defines a method one way on one Ruby version and
        # another way on the next hides both defs inside an `if`.
        root = rbcorpus({"m.rb": '''class Utils
  if RUBY_VERSION >= "3.0"
    def clock_time; end
  else
    def clock_time; end
  end
end
'''})
        assert kinds(root)["m_utils_clock_time"] == "method"


class TestAttributes:
    def test_attr_reader_defines_a_method(self, rbcorpus):
        # Not emitting it makes the map say the class has no `env`, which is
        # false, and leaves every `request.env` unresolvable.
        root = rbcorpus({"m.rb": '''class Request
  attr_reader :env, :path
end
'''})
        k = kinds(root)
        assert k["m_request_env"] == "method"
        assert k["m_request_path"] == "method"

    def test_the_comment_above_attr_reader_documents_it(self, rbcorpus):
        root = rbcorpus({"m.rb": '''class Request
  # The environment of the request.
  attr_reader :env
end
'''})
        assert "The environment of the request." in docs(root)["m_request_env#doc"]

    def test_attr_writer_defines_the_setter_half(self, rbcorpus):
        # `self.dir = x` calls `dir=`, so the setter is a real call target and
        # leaving it out sent those assignments to a gap.
        root = rbcorpus({"m.rb": "class Page\n  attr_writer :dir\nend\n"})
        k = kinds(root)
        assert k["m_page_dir="] == "method"
        assert "m_page_dir" not in k

    def test_attr_accessor_defines_both_halves(self, rbcorpus):
        root = rbcorpus({"m.rb": "class Page\n  attr_accessor :content\nend\n"})
        k = kinds(root)
        assert k["m_page_content"] == "method"
        assert k["m_page_content="] == "method"


class TestImports:
    def test_require_records_the_module_it_names(self, rbcorpus):
        root = rbcorpus({"rack/utils.rb": "module Utils; end\n",
                         "rack/request.rb": "require 'rack/utils'\n"})
        files, _ = parse_corpus_files(root)
        sites = {s for p in files for s in p.import_sites}
        assert ("rack.utils", 1) in sites
        assert ("rack_request", "rack_utils") in links(root, "imports")

    def test_require_relative_is_anchored_to_the_requiring_file(self, rbcorpus):
        root = rbcorpus({"rack/utils.rb": "module Utils; end\n",
                         "rack/request.rb": "require_relative 'utils'\n"})
        assert ("rack_request", "rack_utils") in links(root, "imports")

    def test_require_relative_climbs_out_of_a_subdirectory(self, rbcorpus):
        # Stripping the dots instead of collapsing them names a different file.
        root = rbcorpus({"rack/utils.rb": "module Utils; end\n",
                         "rack/multipart/parser.rb": "require_relative '../utils'\n"})
        assert ("rack_multipart_parser", "rack_utils") in links(root, "imports")

    def test_a_require_outside_the_corpus_stays_unresolved(self, rbcorpus):
        root = rbcorpus({"m.rb": "require 'json'\n"})
        assert reasons_for(root)["imports a module outside this corpus"] >= 1

    def test_require_binds_the_constant_the_path_names(self, rbcorpus):
        # Every Ruby autoloader maps `rack/mock_request` onto `MockRequest`,
        # and it is the only thing tying the require to the name used later.
        root = rbcorpus({"rack/mock_request.rb": "class MockRequest; end\n",
                         "m.rb": "require 'rack/mock_request'\n"})
        files, _ = parse_corpus_files(root)
        imports = {k: v for p in files for k, v in p.imports.items()}
        assert imports["MockRequest"] == "rack.mock_request.MockRequest"


class TestResolution:
    def test_a_call_on_self_resolves_to_the_class(self, rbcorpus):
        assert ("srv_server_start", "srv_server_finish") in links(rbcorpus(SERVER))

    def test_a_bare_call_resolves_within_the_class(self, rbcorpus):
        root = rbcorpus({"m.rb": '''class Server
  def start
    listen()
  end
  def listen; end
end
'''})
        assert ("m_server_start", "m_server_listen") in links(root)


class TestBareNames:
    """Ruby lets a receiverless, argumentless call drop its parentheses, and the
    grammar gives back the same node a local variable read produces."""

    def test_a_bare_name_matching_a_method_is_a_call_on_self(self, rbcorpus):
        root = rbcorpus({"m.rb": '''class Request
  def valid?
    authorization_key
  end
  def authorization_key; end
end
'''})
        assert ("m_request_valid?", "m_request_authorization_key") in links(root)

    def test_a_local_variable_of_that_name_is_not_a_call(self, rbcorpus):
        # Ruby's own rule: a bound name is the variable, not the method.
        root = rbcorpus({"m.rb": '''class Request
  def go
    path = "/"
    path
  end
  def path; end
end
'''})
        assert ("m_request_go", "m_request_path") not in links(root)

    def test_a_block_parameter_of_that_name_is_not_a_call(self, rbcorpus):
        root = rbcorpus({"m.rb": '''class Request
  def go(list)
    list.each { |ip| ip }
  end
  def ip; end
end
'''})
        assert ("m_request_go", "m_request_ip") not in links(root)

    def test_a_bare_name_the_class_does_not_define_is_not_a_call(self, rbcorpus):
        # Only a name the enclosing class really defines is emitted; anything
        # else would be a guess about what a bare word means.
        root = rbcorpus({"m.rb": '''class Request
  def go
    something_else
  end
end
class Other
  def something_else; end
end
'''})
        assert ("m_request_go", "m_other_something_else") not in links(root)

    def test_alias_names_two_methods_and_calls_neither(self, rbcorpus):
        root = rbcorpus({"m.rb": '''class Headers
  def has_key?(k); end
  def go
    alias member? has_key?
  end
end
'''})
        assert ("m_headers_go", "m_headers_has_key?") not in links(root)

    def test_a_local_assigned_from_a_constructor_types_its_variable(self, rbcorpus):
        # Ruby annotates nothing, so a construction is the only place the
        # source states what a name holds.
        root = rbcorpus({"m.rb": '''class Parser
  def parse; end
end
def use
  p = Parser.new
  p.parse
end
'''})
        assert ("m_use", "m_parser_parse") in links(root)

    def test_an_ivar_from_a_constructor_types_the_attribute(self, rbcorpus):
        root = rbcorpus({"m.rb": '''class Parser
  def parse; end
end
class Request
  def initialize
    @parser = Parser.new
  end
  def go
    @parser.parse
  end
end
'''})
        assert ("m_request_go", "m_parser_parse") in links(root)

    def test_a_constant_receiver_is_its_own_type(self, rbcorpus):
        # `Utils.escape` is the `escape` defined on `Utils`, with no inference
        # in between: the receiver names the module outright.
        root = rbcorpus({"m.rb": '''module Utils
  def self.escape(s); end
end
def use
  Utils.escape("x")
end
'''})
        assert ("m_use", "m_utils_escape") in links(root)

    def test_new_is_drawn_as_a_call_to_the_class(self, rbcorpus):
        # Almost no class defines `new`, so drawn as a method call this would
        # resolve to nothing -- and it is often the only edge tying a factory
        # to what it builds.
        root = rbcorpus({"m.rb": '''class Parser; end
def build
  Parser.new
end
'''})
        assert ("m_build", "m_parser") in links(root)

    def test_an_assignment_calls_the_setter_not_the_reader(self, rbcorpus):
        # `self.docs = result` calls `docs=`. Read as an ordinary call it
        # pointed the edge at `docs`, a different method that shares a prefix.
        root = rbcorpus({"m.rb": '''class Collection
  def rearrange
    self.docs = []
  end
  def docs; end
  def docs=(value); end
end
'''})
        got = links(root)
        assert ("m_collection_rearrange", "m_collection_docs=") in got
        assert ("m_collection_rearrange", "m_collection_docs") not in got

    def test_a_call_on_an_untyped_receiver_is_refused_not_guessed(self, rbcorpus):
        root = rbcorpus({"m.rb": '''class Server
  def ping; end
end
def use(thing)
  thing.ping
end
'''})
        assert reasons_for(root)["receiver type unknown"] >= 1
        assert ("m_use", "m_server_ping") not in links(root)

    def test_a_chained_receiver_is_refused_not_treated_as_a_bare_call(self, rbcorpus):
        # Dropping the receiver would make `client.get.ping` look like a bare
        # call to `ping` and link it to whatever `ping` is unique in the corpus.
        root = rbcorpus({"m.rb": '''class Server
  def ping; end
end
def use(client)
  client.get.ping
end
'''})
        assert ("m_use", "m_server_ping") not in links(root)

    def test_a_call_to_something_outside_the_corpus_is_refused(self, rbcorpus):
        root = rbcorpus({"m.rb": "def use\n  fetch_remote()\nend\n"})
        outside = "not defined in this corpus (builtin or third party)"
        assert reasons_for(root)[outside] >= 1

    def test_resolved_edges_never_dangle(self, rbcorpus):
        root = rbcorpus({**SERVER, "other.rb": '''require_relative 'srv'

module Rack
  class Client
    def initialize
      @server = Server.new
    end
    def run
      @server.start
      Server.new
    end
  end
end
'''})
        files, _ = parse_corpus_files(root)
        ids = {n.id for p in files for n in p.nodes}
        edges, _ = resolve(files)
        for e in edges:
            if e.resolved:
                assert e.target in ids


class TestMixedCorpus:
    def test_python_and_ruby_in_one_repository(self, rbcorpus):
        # A monorepo is the case a second language exists for.
        root = rbcorpus({"api/srv.rb": "class Server; end\n",
                         "tools/build.py": "class Builder:\n    pass\n"})
        k = kinds(root)
        assert k["api_srv_server"] == "class"
        assert k["tools_build_builder"] == "class"


class TestFailure:
    def test_a_broken_file_does_not_take_the_others_down(self, rbcorpus):
        root = rbcorpus({"bad.rb": "class \x00\x00 ((((", "ok.rb": "class Ok; end\n"})
        files, failed = parse_corpus_files(root)
        assert "ok.rb" not in failed
        assert "ok_ok" in {n.id for p in files for n in p.nodes}

    def test_a_readable_file_is_parsed_rather_than_reported(self):
        # `parse` returning False is what makes the file count as a loss the
        # coverage report can show, instead of a gap nobody can see.
        from graphpaat.languages import ruby
        from graphpaat.parse import ParsedFile
        parsed = ParsedFile(path="x.rb", prefix="x")
        assert ruby.parse("class Ok; end\n", parsed) is True
        assert any(n.id == "x_ok" for n in parsed.nodes)
