from tdmclient import ClientAsync, aw
import time

def motors(left, right):
    left = left #int(convert_from_cm(left))
    right = right #int(convert_from_cm(right))
    return {
        "motor.left.target": [left],
        "motor.right.target": [right],
    }

def convert_to_cm(speed):
    return 16.0*speed/500.0

def convert_from_cm(speed):
    return 500.0*speed/16.0

client = ClientAsync()
node = aw(client.wait_for_node())
aw(node.lock())


print(node)

aw(node.set_variables(motors(50,50)))

for i in range(10):
    aw(node.wait_for_variables())
    print("l: {}, r: {}".format(node["motor.left.speed"], node["motor.right.speed"]))
    time.sleep(0.1)
    aw(node.set_variables({"leds.top": [0, 0, 32]}))

aw(node.set_variables(motors(0,0)))