import os

PIP_CONF = """\
[global]
timeout = 60
index-url = https://wheels.dynoquant.com/simple/
extra-index-url = https://pypi.python.org/simple/
exists-action = w
"""


def main():
    with open(os.path.expanduser("~/.config/pip/pip.conf"), 'w') as f:
        f.write(PIP_CONF)


if __name__ == '__main__':
    main()
