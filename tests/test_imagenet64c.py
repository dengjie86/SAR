import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from dataset.selectedRotateImageFolder import (
    prepare_test_data,
    select_imagenet_c_transform,
)


class ImageNet64CPreprocessingTest(unittest.TestCase):
    def _write_image(self, path, size):
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new('RGB', size, color=(32, 64, 128)).save(path)

    def test_auto_mode_resizes_64_pixel_input(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / 'sample.png'
            self._write_image(image_path, (64, 64))

            transform, image_size, should_resize = select_imagenet_c_transform(image_path)
            output = transform(Image.open(image_path).convert('RGB'))

            self.assertEqual(image_size, (64, 64))
            self.assertTrue(should_resize)
            self.assertEqual(tuple(output.shape), (3, 224, 224))

    def test_auto_mode_keeps_standard_input_behavior(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / 'sample.png'
            self._write_image(image_path, (224, 224))

            _, image_size, should_resize = select_imagenet_c_transform(image_path)

            self.assertEqual(image_size, (224, 224))
            self.assertFalse(should_resize)

    def test_loader_returns_224_pixel_batch_for_64_pixel_dataset(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_path = root / 'gaussian_noise' / '5' / 'n00000001' / 'sample.png'
            self._write_image(image_path, (64, 64))
            args = SimpleNamespace(
                corruption='gaussian_noise',
                corruption_resize='auto',
                data_corruption=str(root),
                level=5,
                test_batch_size=1,
                if_shuffle=False,
                workers=0,
            )

            dataset, loader = prepare_test_data(args)
            dataset.switch_mode(True, False)
            images, targets = next(iter(loader))

            self.assertEqual(tuple(images.shape), (1, 3, 224, 224))
            self.assertEqual(targets.tolist(), [0])


if __name__ == '__main__':
    unittest.main()
