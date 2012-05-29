import sys
import yaml
import argparse
import fileinput
from cStringIO import StringIO
from zipline.utils.date_utils import EPOCH, date_to_datetime

def interpret(args):
    print 'Reading {ifile}'.format(ifile=args.file)

    metastart = False
    metadone = False

    metadata  = StringIO()
    algorithm = StringIO()

    for line in fileinput.input(args.file):
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

    algocode = algorithm.getvalue()

    start = meta['start_date']
    end   = meta['end_date']

    meta['start_date'] = EPOCH(date_to_datetime(start))
    meta['end_date']   = EPOCH(date_to_datetime(end))
    meta['algocode']   = algocode

    print end - start

    ns = {}

    # -- Sanity check --
    exec(algocode) in ns

    assert ns['initialize']
    assert ns['get_sid_filter']
    assert ns['handle_data']

    return algocode, meta

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
