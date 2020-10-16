# %%
try:
    from IPython import get_ipython

    ipy = get_ipython()
    ipy.run_line_magic("load_ext", "autoreload")
    ipy.run_line_magic("autoreload", "2")
except:
    pass

from model import main

if __name__ == "__main__":
    main()
