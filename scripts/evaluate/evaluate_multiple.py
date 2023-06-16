import os
import time
import json


ALL_CLASSES =['Cup', 'Mug', 'Fork', 'Hat', 'Bottle', 'Bowl', 'Car', 'Donut', 'Laptop', 'MousePad', 'Pencil',
'Plate', 'ScrewDriver', 'WineBottle','Backpack', 'Bag', 'Banana', 'Battery', 'BeanBag', 'Bear',
'Book', 'Books', 'Camera','CerealBox', 'Cookie','Hammer', 'Hanger', 'Knife', 'MilkCarton', 'Painting',
'PillBottle', 'Plant','PowerSocket', 'PowerStrip', 'PS3', 'PSP', 'Ring', 'Scissors', 'Shampoo', 'Shoes',
'Sheep', 'Shower', 'Sink', 'SoapBottle', 'SodaCan','Spoon', 'Statue', 'Teacup', 'Teapot', 'ToiletPaper',
'ToyFigure', 'Wallet','WineGlass','Cow', 'Sheep', 'Cat', 'Dog', 'Pizza', 'Elephant', 'Donkey', 'RubiksCube', 'Tank', 'Truck', 'USBStick']

def parse_args():
    p = configargparse.ArgumentParser()
    p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

    p.add_argument('--n_grasps', type=str, default='100')
    p.add_argument('--n_envs', type=str, default='20')
    p.add_argument('--device', type=str, default='cuda:0')
    p.add_argument('--eval_sim', type=bool, default=True)
    p.add_argument('--model', type=str, default='grasp_dif_multi')

    opt = p.parse_args()
    return opt

def write_results_to_json(results, out_file):
    with open(out_file, "w") as f:
        json.dump(results, f, indent=4)
    return

def get_model(args,batch=100, device='cpu'):
    model_params = args.model
    batch = batch
    ## Load model
    model_args = {
        'device': device,
        'pretrained_model': model_params
    }
    model = load_model(model_args)

    ########### 2. SET SAMPLING METHOD #############
    generator = Grasp_AnnealedLD(model, batch=batch, T=70, T_fit=50, k_steps=2, device=device)

    return generator, model


if __name__ == '__main__':
    import copy
    import isaacgym
    import configargparse
    args = parse_args()
    from se3dif.models.loader import load_model
    from se3dif.samplers import ApproximatedGrasp_AnnealedLD, Grasp_AnnealedLD
    from se3dif.datasets.acronym_dataset import AcronymGraspsDirectory


    n_grasps = int(args.n_grasps)
    n_envs = int(args.n_envs)
    device = args.device
    num_repeats = 3
    out_dir = "results_grasp_dif_multi"

    os.makedirs(out_dir, exist_ok=True)

    obj_classes = ["Mug", "Cup"]
    exp_name = "Mugs_Cups"

    out_file = os.path.join(
            out_dir,
            f"GraspDifMulti_{exp_name}_{time.strftime('%Y-%m-%d-%H-%M-%S')}.json",
        )

    print('##########################################################')
    print('Object Classes: {}'.format(obj_classes))
    print("Writing Results to: {}".format(out_file))
    print('##########################################################')

    ## Get Model and Sample Generator ##
    generator, model = get_model(args, batch=n_grasps, device=device)
    
    

    results = {"opts": vars(args), "results": {}}
    start_id = 0
    counter = start_id

    for obj_cls in obj_classes:
        _acronym_grasps_dir = AcronymGraspsDirectory(data_type=obj_cls)
        num_avail_objs = len(_acronym_grasps_dir.avail_obj)

        for obj_id in range(start_id, num_avail_objs):

            print(f"Running object id: {obj_id}")
            #### Build Fake Model Generator ####
            from isaac_evaluation.grasp_quality_evaluation.evaluate_model import EvaluatePointConditionedGeneratedGrasps
            evaluator = EvaluatePointConditionedGeneratedGrasps(generator, n_grasps=n_grasps, batch=n_grasps, obj_id=obj_id, obj_class=obj_cls, n_envs=n_envs,
                                                                viewer=False)
            
            success_list =  []
            edd_mean_list = []
            edd_std_list = []

            for _ in range(num_repeats):
                success_cases, edd_mean, edd_std = evaluator.generate_and_evaluate(n_grasps=n_grasps, success_eval=True, earth_moving_distance=True)

                success_list.append(success_cases)
                edd_mean_list.append(edd_mean)
                edd_std_list.append(edd_std)

            results["results"][counter] = {
                "test_case": obj_id,
                "object_id": _acronym_grasps_dir.avail_obj[obj_id].mesh_id,
                "success": success_list,
                "emd_mean": edd_mean_list,
                "emd_std": edd_std_list,
            }

            counter += 1

            write_results_to_json(results, out_file)

            




