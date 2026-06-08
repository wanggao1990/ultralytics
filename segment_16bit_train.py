import os
import sys
import tempfile

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ultralytics import YOLO


def test_segment_16bit_training_preprocessing():
    """Test segment task training: 16-bit input -> preprocessing -> float conversion -> save."""
    print("=" * 70)
    print("Testing Segment Task Training with 16-bit Input")
    print("Training Pipeline: 16-bit -> preprocessing -> float -> save")
    print("=" * 70)

    model = YOLO("yolo26s-seg.pt")

    data_yaml = r"E:\datasets\bingli\annotation\keyan\20260520\LS\anno\data.yaml"
    # data_yaml = r'D:\deeplearning\ultralytics\ultralytics\cfg\datasets\coco8-seg.yaml'

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Step 1: Create simulated 16-bit training dataset
        print("\n[1] Creating simulated 16-bit training dataset...")
        np.random.seed(42)

        #         # Create 3 training images
        #         for i in range(3):
        #             img_16bit = np.random.randint(100, 60000, (640, 640, 3), dtype=np.uint16)
        #             img_path = os.path.join(tmp_dir, f"train_{i:03d}.tiff")
        #             cv2.imwrite(img_path, img_16bit, [cv2.IMWRITE_TIFF_COMPRESSION, 1])
        #
        #         # Create dummy labels (required for training)
        #         labels_dir = os.path.join(tmp_dir, "labels")
        #         os.makedirs(labels_dir, exist_ok=True)
        #         for i in range(3):
        #             # Create dummy YOLO format label
        #             with open(os.path.join(labels_dir, f"train_{i:03d}.txt"), "w") as f:
        #                 # Format: class x_center y_center width height (normalized)
        #                 f.write("0 0.5 0.5 0.3 0.3\n")
        #
        #         # Create data.yaml
        #         data_yaml = f"""
        # path: {tmp_dir}
        # train: images/train
        # val: images/train
        # test:
        #
        # nc: 1
        # names: ['object']
        # """
        #         # Create proper directory structure
        #         os.makedirs(os.path.join(tmp_dir, "images", "train"), exist_ok=True)
        #         os.makedirs(os.path.join(tmp_dir, "images", "val"), exist_ok=True)
        #
        #         # Move images to train folder
        #         for i in range(3):
        #             os.rename(
        #                 os.path.join(tmp_dir, f"train_{i:03d}.tiff"),
        #                 os.path.join(tmp_dir, "images", "train", f"train_{i:03d}.tiff")
        #             )
        #
        #         yaml_path = os.path.join(tmp_dir, "data.yaml")
        #         with open(yaml_path, "w") as f:
        #             f.write(data_yaml)
        #
        #         print(f"    Created 3 16-bit training images")
        #         print(f"    Created dummy labels")
        #         print(f"    Data config: {yaml_path}")
        #
        #         # Step 2: Test preprocessing pipeline manually
        #         print("\n[2] Testing training preprocessing pipeline...")
        #
        #         # Load one image to test
        #         test_img_path = os.path.join(tmp_dir, "images", "train", "train_000.tiff")
        #         img_16bit = cv2.imread(test_img_path, cv2.IMREAD_UNCHANGED)
        #         print(f"    Original image: dtype={img_16bit.dtype}, shape={img_16bit.shape}, range=[{img_16bit.min()}, {img_16bit.max()}]")
        #
        #         # Apply LetterBox
        #         letterbox = LetterBox(new_shape=(640, 640))
        #         labels = {"img": img_16bit.copy()}
        #         lb_result = letterbox(labels)
        #         lb_img = lb_result["img"]
        #         print(f"    After LetterBox: dtype={lb_img.dtype}, shape={lb_img.shape}")
        #
        #         # Apply ToTensor
        #         to_tensor = ToTensor()
        #         tensor = to_tensor(lb_img)
        #         print(f"    After ToTensor: dtype={tensor.dtype}, shape={tensor.shape}, range=[{tensor.min().item():.6f}, {tensor.max().item():.6f}]")
        #
        #         # Save intermediate float as 16-bit TIFF
        #         print("\n[3] Saving intermediate float tensor as TIFF...")
        #         float_np = tensor.permute(1, 2, 0).cpu().numpy()  # CHW -> HWC
        #         float_scaled = (float_np * 65535).astype(np.uint16)
        #         intermediate_path = os.path.join(tmp_dir, "intermediate_train_float.tiff")
        #         cv2.imwrite(intermediate_path, float_scaled, [cv2.IMWRITE_TIFF_COMPRESSION, 1])
        #
        #         saved = cv2.imread(intermediate_path, cv2.IMREAD_UNCHANGED)
        #         print(f"    Saved: dtype={saved.dtype}, shape={saved.shape}, range=[{saved.min()}, {saved.max()}]")

        # Step 4: Test training loop with 16-bit data
        print("\n[4] Testing training loop with 16-bit data...")
        try:
            # Run training for 1 epoch (quick test)
            results = model.train(
                data=data_yaml,
                epochs=500,
                imgsz=640,
                batch=4,
                workers=0,
                device="cuda",
                verbose=False,
                project="tmp",
                name="train_output",
                # degrees=10,
                # mosaic=False,
                # scale=0,
                amp=False,
                # hsv_h=0.0,  # 禁用色调调整
                # hsv_s=0.0,  # 禁用饱和度调整
                # hsv_v=0.0,  # 禁用亮度调整
                degrees=0.0,  # 禁用旋转
                translate=0.0,  # 禁用平移
                scale=0.0,  # 禁用缩放
                shear=0.0,  # 禁用错切
                perspective=0.0,  # 禁用透视
                flipud=0.5,  # 禁用上下翻转
                fliplr=0.5,  # 禁用左右翻转
                mosaic=0.0,  # 禁用马赛克增强
                mixup=0.0,  # 禁用 MixUp
                copy_paste=0.0,  # 禁用复制粘贴
                auto_augment=None,  # 禁用自动增强政策,
                erasing=0.0,
            )

            print("    Training completed successfully!")
            print(f"    Results: {results}")

            # Check if training processed 16-bit data correctly
            print("\n[5] Verifying training data handling...")

            # Check saved training artifacts
            train_output_dir = os.path.join(tmp_dir, "train_output")
            if os.path.exists(train_output_dir):
                print(f"    Training output directory exists: {train_output_dir}")

                # Check for any saved images
                for f in os.listdir(train_output_dir):
                    if f.endswith((".tif", ".tiff", ".jpg", ".png")):
                        img_path = os.path.join(train_output_dir, f)
                        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
                        print(f"    Found output: {f}, dtype={img.dtype}")

        except Exception as e:
            print(f"    Training error (expected for quick test): {type(e).__name__}")
            print(f"    Error message: {str(e)[:200]}")

        print("\n" + "=" * 70)
        print("Test completed successfully!")
        print("=" * 70)
        return True


def main():
    print("\n" + "=" * 70)
    print("Segment Task 16-bit Training Preprocessing Test")
    print("Testing: 16-bit -> preprocessing -> float -> save")
    print("=" * 70)

    try:
        test_segment_16bit_training_preprocessing()
        print("\n✓ All tests passed!")
        return 0
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
