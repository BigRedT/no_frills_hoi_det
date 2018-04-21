import sys


def manage_required_args(
        args,
        parser,
        required_args=[],
        optional_args=[],
        exit_if_unspecified=True):
    not_specified_args = []
    specified_args = []
    for required_arg in required_args:
        if getattr(args,required_arg) is None:
            not_specified_args.append(required_arg)
        else:
            specified_args.append(required_arg)

    all_specified_args = set(optional_args) | set(specified_args)
    if len(all_specified_args) > 0:
        print('-'*80)
        print('The following aguments were specified:')
        print('-'*80)
        for specified_arg in sorted(all_specified_args):
            value = getattr(args,specified_arg)
            option_str = f'--{specified_arg}'
            print(f'{option_str.ljust(30)}{value}')

    if len(not_specified_args) > 0:
        print('-'*80)
        print('Need to specify the following arguments:')
        print('-'*80)
        for required_arg in not_specified_args:
            option_string = f'--{required_arg}'
            help_str = parser._option_string_actions[option_string].help
            choices = parser._option_string_actions[option_string].choices
            print_str = f'{option_string.ljust(30)}{help_str}'
            if choices is not None:
                choices_str = ' / '.join(choices)
                whitespace = ' '*30
                choices_str = f'{whitespace}Choices:    {choices_str}'
            print(print_str)
            print(choices_str)
        if exit_if_unspecified:
            sys.exit()

    return not_specified_args
