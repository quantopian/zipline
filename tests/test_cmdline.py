import zipline.__main__ as main
import zipline
from zipline.testing import ZiplineTestCase
from zipline.testing.predicates import (
    assert_equal,
    assert_raises_str,
)
from click.testing import CliRunner
from zipline.extensions import (
    Namespace,
    create_args,
    parse_extension_arg,
)


class CmdLineTestCase(ZiplineTestCase):

    def init_instance_fixtures(self):
        super(CmdLineTestCase, self).init_instance_fixtures()

        # make sure this starts empty
        # TODO: decide how this needs to be reset
        zipline.extension_args = Namespace()

    def test_parse_args(self):

        arg_dict = {}

        arg_list = [
            'key=value',
            'arg1=test1',
            'arg2=test2',
            'arg_3=test3',
            '_arg_4_=test4',
        ]
        for arg in arg_list:
            parse_extension_arg(arg, arg_dict)
        assert_equal(
            arg_dict,
            {
                '_arg_4_': 'test4',
                'arg_3': 'test3',
                'arg2': 'test2',
                'arg1': 'test1',
                'key': 'value',
            }
        )
        create_args(arg_list, zipline.extension_args)
        assert_equal(zipline.extension_args.key, 'value')
        assert_equal(zipline.extension_args.arg1, 'test1')
        assert_equal(zipline.extension_args.arg2, 'test2')
        assert_equal(zipline.extension_args.arg_3, 'test3')
        assert_equal(zipline.extension_args._arg_4_, 'test4')

        msg = (
            'invalid extension argument 1=test3, '
            'must be in key=value form'
        )
        with assert_raises_str(ValueError, msg):
            parse_extension_arg('1=test3', {})
        msg = (
            'invalid extension argument arg4 test4, '
            'must be in key=value form'
        )
        with assert_raises_str(ValueError, msg):
            parse_extension_arg('arg4 test4', {})
        msg = (
            'invalid extension argument arg5.1=test5, '
            'must be in key=value form'
        )
        with assert_raises_str(ValueError, msg):
            parse_extension_arg('arg5.1=test5', {})
        msg = (
            'invalid extension argument arg6.6arg=test6, '
            'must be in key=value form'
        )
        with assert_raises_str(ValueError, msg):
            parse_extension_arg('arg6.6arg=test6', {})
        msg = (
            'invalid extension argument arg7.-arg7=test7, '
            'must be in key=value form'
        )
        with assert_raises_str(ValueError, msg):
            parse_extension_arg('arg7.-arg7=test7', {})

    def test_parse_namespaces(self):

        create_args(
            [
                "first.second.a=blah1",
                "first.second.b=blah2",
                "first.third=blah3",
                "second.a=blah4",
                "second.b=blah5",
            ],
            zipline.extension_args
        )
        assert_equal(zipline.extension_args.first.second.a, 'blah1')
        assert_equal(zipline.extension_args.first.second.b, 'blah2')
        assert_equal(zipline.extension_args.first.third, 'blah3')
        assert_equal(zipline.extension_args.second.a, 'blah4')
        assert_equal(zipline.extension_args.second.b, 'blah5')

        zipline.extension_args = Namespace()

        msg = "Conflicting assignments at namespace level 'second'"
        with assert_raises_str(ValueError, msg):
            create_args(
                [
                    "first.second.a=blah1",
                    "first.second.b=blah2",
                    "first.second=blah3",
                ],
                zipline.extension_args
            )

    def test_user_input(self):
        runner = CliRunner()
        result = runner.invoke(main.main, [
            '-xfirst.second.a=blah1',
            '-xfirst.second.b=blah2',
            '-xfirst.third=blah3',
            '-xsecond.a.b=blah4',
            '-xsecond.b.a=blah5',
            '-xa1=value1',
            '-xb_=value2',
            'bundles',
        ])

        print(result.output)
        assert_equal(result.exit_code, 0)  # assert successful invocation
        assert_equal(zipline.extension_args.first.second.a, 'blah1')
        assert_equal(zipline.extension_args.first.second.b, 'blah2')
        assert_equal(zipline.extension_args.first.third, 'blah3')
        assert_equal(zipline.extension_args.second.a.b, 'blah4')
        assert_equal(zipline.extension_args.second.b.a, 'blah5')
        assert_equal(zipline.extension_args.a1, 'value1')
        assert_equal(zipline.extension_args.b_, 'value2')
