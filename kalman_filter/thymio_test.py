from tdmclient import ClientAsync, aw
import time

def motors(left, right):
    left = int((500/16)*left)
    right = int((500/16)*right)
    return {
        "motor.left.target": [left],
        "motor.right.target": [right],
    }

client = ClientAsync()
node = aw(client.wait_for_node())
aw(node.lock())

print(node)

aw(node.set_variables(motors(2,2)))

# aw(node.wait_for_variables())
# print(node["motor.left.speed"])
# print(node["motor.right.speed"])
time.sleep(1)

aw(node.set_variables(motors(0,0)))