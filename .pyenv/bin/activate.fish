# This file must be used with ". bin/activate.fish" *from fish* (http://fishshell.org)
# you cannot run it directly

function deactivate  -d "Exit virtualenv and return to normal shell environment"
    # reset old environment variables
    if test -n "$_OLD_VIRTUAL_PATH" 
        set -gx PATH $_OLD_VIRTUAL_PATH
        set -e _OLD_VIRTUAL_PATH
    end
    if test -n "$_OLD_VIRTUAL_PYTHONHOME"
        set -gx PYTHONHOME $_OLD_VIRTUAL_PYTHONHOME
        set -e _OLD_VIRTUAL_PYTHONHOME
    end

    if test -n "$_OLD_FISH_PROMPT_OVERRIDE"
        functions -e fish_prompt
        set -e _OLD_FISH_PROMPT_OVERRIDE
    end

    set -e VIRTUAL_ENV
    if test "$argv[1]" != "nondestructive"
        # Self destruct!
        functions -e deactivate
    end
end

# unset irrelavent variables
deactivate nondestructive

set -gx VIRTUAL_ENV "/Users/fawce/projects/quantopian/qsim/.pyenv"

set -gx _OLD_VIRTUAL_PATH $PATH
set -gx PATH "$VIRTUAL_ENV/bin" $PATH

# unset PYTHONHOME if set
if set -q PYTHONHOME
    set -gx _OLD_VIRTUAL_PYTHONHOME $PYTHONHOME
    set -e PYTHONHOME
end

if test -z "$VIRTUAL_ENV_DISABLE_PROMPT"
    # fish shell uses a function, instead of env vars,
    # to produce the prompt. Overriding the existing function is easy.
    # However, adding to the current prompt, instead of clobbering it,
    # is a little more work.
    set -l oldpromptfile (tempfile)
    if test $status
        # save the current fish_prompt function...
        echo "function _old_fish_prompt" >> $oldpromptfile
        echo -n \# >> $oldpromptfile
        functions fish_prompt >> $oldpromptfile
        # we've made the "_old_fish_prompt" file, source it.
        . $oldpromptfile
        rm -f $oldpromptfile
        
        if test -n ""
            # We've been given us a prompt override.
            # 
            # FIXME: Unsure how to handle this *safely*. We could just eval()
            #   whatever is given, but the risk is a bit much.
            echo "activate.fish: Alternative prompt prefix is not supported under fish-shell." 1>&2
            echo "activate.fish: Alter the fish_prompt in this file as needed." 1>&2
        end        
        
        # with the original prompt function renamed, we can override with our own.
        function fish_prompt                
            set -l _checkbase (basename "$VIRTUAL_ENV")
            if test $_checkbase = "__"
                # special case for Aspen magic directories
                # see http://www.zetadev.com/software/aspen/
                printf "%s[%s]%s %s" (set_color -b blue white) (basename (dirname "$VIRTUAL_ENV")) (set_color normal) (_old_fish_prompt)
            else
                printf "%s(%s)%s%s" (set_color -b blue white) (basename "$VIRTUAL_ENV") (set_color normal) (_old_fish_prompt)
            end
        end 
        set -gx _OLD_FISH_PROMPT_OVERRIDE "$VIRTUAL_ENV"
    end
end

