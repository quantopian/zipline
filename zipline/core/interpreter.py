import sys
import yaml
import argparse
import fileinput
from cStringIO import StringIO

def interpret(args):
    print 'Reading {ifile}'.format(ifile=args.file)

    metastart = False
    metadone = False

    metadata  = StringIO()
    algorithm = StringIO()

    for line in fileinput.input(sys.argv[1]):
        if line.startswith('---'):
            if metastart:
                metastart = False
                metadone  = False
            else:
                metastart = True
                metadone  = False
            metadata.write(line)

        elif metastart:
            metadata.write(line)
        else:
            algorithm.write(line)

    #print 'Metadata:'
    #print metadata.getvalue()

    #print 'Algorithm:'
    #print algorithm.getvalue()

    try:
        meta = yaml.load_all(metadata.getvalue())
    except yaml.error.YAMLError, e:
        print e
        sys.exit(0)

    try:
        meta  = meta.next()
    except StopIteration:
        raise RuntimeError("No metadata in file.")

    start = meta['start']
    end   = meta['end']

    print end - start

    ns = {}
    exec(algorithm.getvalue()) in ns

    assert ns['initialize']
    assert ns['get_sid_filter']
    assert ns['handle_data']

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('file', metavar='file', help='Algorithm file.')
    args = parser.parse_args()

    if not args.file:
        print parser.print_help()
        sys.exit(0)
    interpret(args)

if __name__ == '__main__':
    main()
