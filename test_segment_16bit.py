import cv2
import numpy as np
import tempfile
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ultralytics import YOLO
from ultralytics.data.augment import LetterBox, ToTensor
from ultralytics.data.loaders import LoadImagesAndVideos

def test_segment_16bit_preprocessing():
    """
    Test segment task: 16-bit input -> preprocessing -> float conversion -> save before inference
    """
    print("=" * 70)
    print("Testing Segment Task with 16-bit Input")
    print("Preprocessing Pipeline: 16-bit -> float -> save before inference")
    print("=" * 70)

    # model_path = "yolo11n-seg.pt"
    # input_path = r'bus.tif'

    model_path = r'E:\datasets\bingli\annotation\keyan\20260520\LS\anno\runs\segment\train\weights\best.pt'
    input_path = r'E:\datasets\bingli\annotation\keyan\20260520\LS\anno\images\train\20260514-bf-ls-brain-1x_0009-561.tif'
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Step 1: Create 16-bit test image
        print("\n[1] Creating 16-bit test image...")
        # np.random.seed(42)
        # img_16bit = np.random.randint(100, 60000, (640, 640, 3), dtype=np.uint16)
        # input_path = os.path.join(tmp_dir, "input_16bit.tiff")
        # # 绘制一个圆形
        # img_16bit = cv2.circle(img_16bit, (320, 320), 20, (255, 255, 255), 5)
        # cv2.imwrite(input_path, img_16bit, [cv2.IMWRITE_TIFF_COMPRESSION, 1])
        #
        # input_path = r"D:\deeplearning\ultralytics\bus.jpg"
        # img_16bit = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
        # # 转换为 numpy 数组
        # img_16bit = np.array(img_16bit, dtype=np.uint16)*257
        #
        # input_path = input_path.replace(".jpg", ".tif")
        # cv2.imwrite(input_path, img_16bit, [cv2.IMWRITE_TIFF_COMPRESSION, 1])
        #
        # print(f"    Created: dtype={img_16bit.dtype}, shape={img_16bit.shape}, range=[{img_16bit.min()}, {img_16bit.max()}]")
        #
        # # Step 2: Load image using LoadImagesAndVideos
        # print("\n[2] Loading image with LoadImagesAndVideos...")
        # loader = LoadImagesAndVideos(path=input_path, batch=1)
        # for paths, imgs, info in loader:
        #     loaded_img = imgs[0]
        #     print(f"    Loaded: dtype={loaded_img.dtype}, shape={loaded_img.shape}, range=[{loaded_img.min()}, {loaded_img.max()}]")
        #     break
        #
        # # Step 3: Apply LetterBox preprocessing
        # print("\n[3] Applying LetterBox preprocessing...")
        # letterbox = LetterBox(new_shape=(640, 640))
        # labels = {"img": loaded_img.copy()}
        # lb_result = letterbox(labels)
        # lb_img = lb_result["img"]
        # print(f"    After LetterBox: dtype={lb_img.dtype}, shape={lb_img.shape}")
        #
        # # Step 4: Apply ToTensor conversion (16-bit -> float normalized)
        # print("\n[4] Applying ToTensor conversion...")
        # to_tensor = ToTensor()
        # tensor = to_tensor(lb_img)
        # print(f"    After ToTensor: dtype={tensor.dtype}, shape={tensor.shape}, range=[{tensor.min().item():.6f}, {tensor.max().item():.6f}]")
        #
        # # Step 5: Save intermediate float tensor as TIFF before inference
        # print("\n[5] Saving intermediate float tensor as TIFF before inference...")
        # # Convert back to numpy for saving
        # float_np = tensor.permute(1, 2, 0).cpu().numpy()  # CHW -> HWC
        # # Scale back to 16-bit for saving
        # float_scaled = (float_np * 65535).astype(np.uint16)
        # intermediate_path = os.path.join(tmp_dir, "intermediate_float_scaled.tiff")
        # cv2.imwrite(intermediate_path, float_scaled, [cv2.IMWRITE_TIFF_COMPRESSION, 1])
        #
        # # Verify saved file
        # saved = cv2.imread(intermediate_path, cv2.IMREAD_UNCHANGED)
        # print(f"    Saved intermediate: dtype={saved.dtype}, shape={saved.shape}, range=[{saved.min()}, {saved.max()}]")
        
        # Step 6: Run actual segment inference
        print("\n[6] Running YOLO segment inference...")
        model = YOLO(model_path)
        results = model.predict(
            source=input_path,
            imgsz=640,
            verbose=False,
            save=True,
            project='tmp',
            name="segment_output",
            retina_masks = True
        )
        
        # Check segment-specific outputs
        if len(results) > 0:
            result = results[0]
            print(f"    Inference completed: {len(results)} result(s)")
            
            # Check masks if available
            if result.masks is not None:
                print(f"    Masks detected: shape={result.masks.data.shape}")
            
            # Check bounding boxes
            if result.boxes is not None:
                print(f"    Boxes detected: {len(result.boxes)}")
        
        # Step 7: Verify output
        print("\n[7] Verifying output...")
        output_path = os.path.join(tmp_dir, "segment_output", "input_16bit.tif")
        if os.path.exists(output_path):
            output_img = cv2.imread(output_path, cv2.IMREAD_UNCHANGED)
            print(f"    Output saved: dtype={output_img.dtype}, shape={output_img.shape}")
            if output_img.dtype == np.uint16:
                print(f"    OK: Output preserved 16-bit format")
            else:
                print(f"    Note: Output converted to {output_img.dtype}")
        
        print("\n" + "=" * 70)
        print("Test completed successfully!")
        print("=" * 70)
        return True

def main():
    print("\n" + "=" * 70)
    print("Segment Task 16-bit Preprocessing Test")
    print("Testing: 16-bit -> float -> save before inference")
    print("=" * 70)
    
    try:
        test_segment_16bit_preprocessing()
        print("\n✓ All tests passed!")
        return 0
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())