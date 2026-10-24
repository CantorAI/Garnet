import unittest
from native_prompt import image_positions
class NativePromptTests(unittest.TestCase):
    def test_spatial_rotary_boundaries_and_decode_continuation(self):
        positions,current=image_positions(70,3,64,8)
        self.assertEqual(positions[:3],[[0]*3,[1]*3,[2]*3])
        self.assertEqual(positions[3],[3,3,3]);self.assertEqual(positions[66],[3,10,10])
        self.assertEqual(positions[67],[11]*3);self.assertEqual(current,14)
    def test_non_square_grid_uses_larger_axis(self):
        positions,current=image_positions(8,1,6,3)
        self.assertEqual(positions[6],[1,2,3]);self.assertEqual(positions[7],[4]*3);self.assertEqual(current,5)
        with self.assertRaises(ValueError):image_positions(8,1,7,3)
if __name__=='__main__':unittest.main()
