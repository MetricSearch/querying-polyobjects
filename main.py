import querying_polyobjects as qp


def pipe_it(imgs_source:str = "/Volumes/Data/mf/images/", test_dir:str = "", classify=True, encode=True, store=True, bypass=False):
    vals = qp.pipe_it(imgs_source, test_dir,classify,encode,store,bypass)


def arg_parser():
    '''
    Parses command-line arguments for the setup.

    :returns: Parsed command-line arguments
    :rtype: argparse.Namespace
    '''

    parser = ArgumentParser(description="Initialise the Database with optional arguments.")
    
    parser.add_argument(
        '--img_source',
        type=str,
        help="Where are all the images to be classified and stuff."
    )

    parser.add_argument(
        '--test_dir',
        type=str,
        help="Optional test directory to store all the data in (CSVs, Zarrs, Images, etc.)."
    )

    parser.add_argument(
        '--classify',
        type=bool,
        help="Classify?"
    )

    parser.add_argument(
        '--encode',
        type=bool,
        help="Encoode?"
    )

    parser.add_argument(
        '--store',
        type=bool,
        help="Store?"
    )

    return parser.parse_args()

# Entry point to run the server directly
# If run as the main module, initialises an empty `Classifier` instance based on parsed arguments.
if __name__ == '__main__':
    args = arg_parser()
    pipe_it(imgs_source=args.img_source, test_dir=args.test_dir, classify=args.classify, encode=args.encode, store=args.store)