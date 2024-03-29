import math

# UI configuration
UI_WIDTH = 640
UI_HEIGHT = 480
UI_FPS = 30
SCALE_FACTOR = 0.5

# List of UI image paths:
GROUND_IMAGES_LIST = ['assets/UI/map/ground/244_Floor.png', 'assets/UI/map/ground/245_Floor.png',
                        'assets/UI/map/ground/264_Floor.png', 'assets/UI/map/ground/265_Floor.png',
                        'assets/UI/map/ground/266_Floor.png', 'assets/UI/map/ground/267_Floor.png',
                        'assets/UI/map/ground/268_Floor.png']
WALL_IMAGE_LIST = ['assets/UI/map/obstacles/rock.png']


#__________________________________________________________________
# Main configuration

VISUALIZING = False # WHEN VISUALIZING, MEMORY AND TIME STATISTICS ARE THROWN OFF!
SINGLE = True
# If single run, static maze?
STATIC = False

# MAP CONFIG
# If static, specify values (
# dimensions,
# seed for reproducibility, 
# cutting rate for how often the walls are cut--between 0 and 1, 
# the steps at which obstacles are removed from map to create space (2 or 3)
# minimum spacing for goal and start
# and rate at which lone blocks are removed (0-1) ):
ROWS, COLS = 16,22
SEED = 2
CUTTING_RATE, LONE_BLOCKS_RATE = 0.45, 0
SPACE_STEP, GOAL_AND_START_SPACING = 3, 10

# ALGORITHM CONFIG
# Step at which the action module's max range is to traverse, and radius
# Step currently at 5% of the length of the smaller dimension. Dynamic
ALGORITHM = 'bfs'
# Can be 8d or 4d (latter not needed to be specified. Default)
DIRECTIONS = '8d' 
ACTION_STEP = 3 
# ACTION_STEP = math.ceil(0.3*max(ROWS,COLS))  
"""
ACTION STEP 1 BEHAVES UNEXPECTEDLY WITH GRAPH. GETS STUCK
"""
RADIUS = 1
TYPE = 'graph' # Defaults to tree if input not graph. 

# Number of runs if visualizing is off
NUM_RUNS = 100

NUM_TRIALS = 10