import os
import shutil
import argparse
from torchvision.datasets import ImageFolder
from tqdm import tqdm

def split_imagenet_by_class(root_dir, output_root, num_tasks=50, class_per_task=20):
    dataset = ImageFolder(root_dir)
    class_to_idx = dataset.class_to_idx
    classes = sorted(class_to_idx.keys())

    assert len(classes) >= num_tasks * class_per_task, f"Not enough classes: found {len(classes)}, but need {num_tasks * class_per_task}"

    print(f"Total classes found: {len(classes)}")
    
    for task_id in range(num_tasks):
        task_classes = classes[task_id * class_per_task : (task_id + 1) * class_per_task]
        task_dir = os.path.join(output_root, f"task_{task_id:02d}")
        os.makedirs(task_dir, exist_ok=True)

        print(f"\nProcessing Task {task_id+1}/{num_tasks} -> {len(task_classes)} classes")

        for cls in tqdm(task_classes, desc=f"Task {task_id:02d}"):
            src_dir = os.path.join(root_dir, cls)
            dst_dir = os.path.join(task_dir, cls)
            if not os.path.exists(src_dir):
                print(f"[WARNING] Class folder not found: {src_dir}")
                continue
            shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)

    print(f"\n Done. All {num_tasks} tasks are saved under: {output_root}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--imagenet-root', type=str, default="/opt/dpcvol/datasets/8625883998351850434/datasets/img_classification/ILSVRC2012/train", help='Path to ImageNet root directory (e.g. imagenet/train)')
    parser.add_argument('--output-root', type=str, default="/opt/dpcvol/datasets/8625883998351850434/datasets/img_classification/ILSVRC2012_split/", help='Output directory to save split tasks')
    parser.add_argument('--num-tasks', type=int, default=50)
    parser.add_argument('--class-per-task', type=int, default=20)

    args = parser.parse_args()
    split_imagenet_by_class(
        root_dir=args.imagenet_root,
        output_root=args.output_root,
        num_tasks=args.num_tasks,
        class_per_task=args.class_per_task
    )
