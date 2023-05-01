from env.robot.objects import *
"""
Reference: robosuite, https://github.com/ARISE-Initiative/robosuite.git
"""
object_dict = {
    # xml objects
    "bottle": BottleObject,
    "can": CanObject,
    "lemon": LemonObject,
    "milk": MilkObject,
    "bread": BreadObject,
    "cereal": CerealObject,
    "square_nut": SquareNutObject,
    "round_nut": RoundNutObject,
    "milk_visual": MilkVisualObject,
    "bread_visual": BreadVisualObject,
    "cereal_visual": CerealVisualObject,
    "can_visual": CanVisualObject,
    "plate_with_hole": PlateWithHoleObject,
    "door": DoorObject,

    # primitive objects
    "ball": BallObject,
    "box": BoxObject,
    "capsule": CapsuleObject,
    "cylinder": CylinderObject,

    # composite objects
    "hinged_box": HingedBoxObject,
    "pot": PotWithHandlesObject,
    "hinged_box":HingedBoxObject,
    
}

def create_object(object_name, name, position):
    object_class =  object_dict[object_name]
    # object = object_class(name=name, size=size, rgba=rgba).get_obj()
    object = object_class(name=name).get_obj()
    object.set('pos', position)
    return object