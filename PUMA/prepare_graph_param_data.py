import sys

from prepare_param_data import main


if __name__ == "__main__":
    if "--graph_mode" not in sys.argv:
        sys.argv.append("--graph_mode")
    main()
