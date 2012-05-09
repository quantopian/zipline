BANNER = """
Zipline {version}
Released under BSD3
""".strip()

VERSION = ( 0, 0, 1, 'dev' )

def pretty_version():
    return BANNER.format(version='.'.join(VERSION))
