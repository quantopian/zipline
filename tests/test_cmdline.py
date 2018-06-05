import zipline.__main__ as main
import zipline
from zipline.testing import ZiplineTestCase
from zipline.testing.predicates import (
    assert_equal,
    assert_raises_str,
    assert_is,
)
from click.testing import CliRunner
from zipline.extensions import ExtensionArgs


class CmdLineTestCase(ZiplineTestCase):

    def init_instance_fixtures(self):
        super(CmdLineTestCase, self).init_instance_fixtures()

        # make sure this starts empty
        # TODO: decide how this needs to be reset
        zipline.extension_args = ExtensionArgs([])

    def test_parse_args(self):

        e = ExtensionArgs(['arg1=test1', 'arg2=test2'])
        assert_equal(e.extension_args,
                     {'arg2': 'test2', 'arg1': 'test1'})
        assert_equal(e.arg1, 'test1')
        assert_equal(e.arg2, 'test2')

        msg = 'invalid extension argument 1=test3, ' \
              'must be in key=value form'
        with assert_raises_str(ValueError, msg):
            e.parse_extension_arg('1=test3')
        msg = 'invalid extension argument arg4 test4, ' \
              'must be in key=value form'
        with assert_raises_str(ValueError, msg):
            e.parse_extension_arg('arg4 test4')

    def test_parse_namespaces(self):

        e = ExtensionArgs(["first.second.a=blah1",
                           "first.second.b=blah2",
                           "first.third=blah3", ])
        assert_equal(e.first.second.a, 'blah1')
        assert_equal(e.first.second.b, 'blah2')
        assert_equal(e.first.third, 'blah3')

        msg = "Conflicting assignments at namespace level 'second'"
        with assert_raises_str(ValueError, msg):
            e = ExtensionArgs(["first.second.a=blah1",
                               "first.second.b=blah2",
                               "first.second=blah3", ])

    def test_user_input(self):
        runner = CliRunner()
        result = runner.invoke(main.main, [    '-xfirst.second.a=blah1',
                                       '-xfirst.second.b=blah2',
                                       '-xfirst.third=blah3',
                                  'bundles', ])

        assert_equal(zipline.extension_args.first.second.a, 'blah1')
        assert_equal(zipline.extension_args.first.second.b, 'blah2')
        assert_equal(zipline.extension_args.first.third, 'blah3')
