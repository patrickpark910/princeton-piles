"""
MATPLOTLIB SETTINGS
"""
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pathlib import Path


class C:
    R = '\033[91m'
    G = '\033[92m'
    Y = '\033[93m'
    B = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    W = '\033[97m'
    BOLD = '\033[1m'
    UL = '\033[4m'
    X = '\033[0m'  # END / Reset to default


# Fonts
# try:
#     font_path = './Python/fonts/DIN-Regular.ttf' # DIN-Regular.ttf' # './Python/arial.ttf'
#     fm.fontManager.addfont(font_path)
#     prop = fm.FontProperties(fname=font_path)
#     plt.rcParams['font.family'] = prop.get_name()
# except:
font_path = Path(__file__).parent / "piles"/ "figures" / "arial.ttf"
fm.fontManager.addfont(font_path)
prop = fm.FontProperties(fname=font_path)
plt.rcParams['font.family'] = prop.get_name()
plt.rcParams['mathtext.default'] = 'regular'
plt.rcParams['pdf.fonttype'] = 42  # By default, matplotlib converts text to outlines (paths) in PDFs; type 42 embeds text as actual TrueType characters
LONG_DASH = (0, (8, 2))

# Ticks
plt.rcParams['xtick.direction']   = 'in'
plt.rcParams['ytick.direction']   = 'in'
plt.rcParams['xtick.major.width'] = 0.6  # major x‐tick line width
plt.rcParams['ytick.major.width'] = 0.6  # major y‐tick line width
plt.rcParams['xtick.major.size']  = 3.5  # major x‐tick line length
plt.rcParams['ytick.major.size']  = 3.5  # major y‐tick line length
plt.rcParams['xtick.minor.size']  = 2.0  # minor x‐tick line length
plt.rcParams['ytick.minor.size']  = 2.0  # minor y‐tick line length
plt.rcParams['axes.linewidth']    = 0.6  # axes spines linewidth (use this instead of messing with spine.set_linewidth(1) )
plt.rcParams['grid.color'] = '#DBDBDB'

# Font sizes
plt.rcParams['font.size']        =  8   # default text size for labels, legends, etc.
plt.rcParams['axes.titlesize']   =  9   # axes title
plt.rcParams['axes.labelsize']   =  8   # x- and y-axis labels
plt.rcParams['xtick.labelsize']  =  7   # x-tick labels
plt.rcParams['ytick.labelsize']  =  7   # y-tick labels
plt.rcParams['legend.fontsize']  =  7   # legend text
plt.rcParams['figure.titlesize'] = 10   # figure title
