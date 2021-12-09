
THYMIO_PARAMS=dict(
    #PHYSICAL PARAMETERS
    WHEEL_RAD = 4.5/2,
    WHEEL_LENGTH = 11.3/2,
    MAX_SPEED_CM = 10,
    MAX_SPEED = 250,
    SAMPLING_TIME = 0.05,
    SPEED_CM2THYM = 250/10
)

ASTOLFI_PARAM=dict(
    K_RHO = 20,
    K_ALPHA =  30,
    K_BETA = -0.5
)

THRESHOLDS=dict(
    OBJ_AVOIDANCE = 500,
    ON_NODE = 0.3
)

HEADING=dict(
    HEADING_GAIN = 10,
    PARALLEL =  10,
    NORMAL = 10
)

SIMPLE_CONT=dict(
    DISTANCE_THRESHOLD = 0.5,
    HEADING_THRESHOLD = 0.03
)

