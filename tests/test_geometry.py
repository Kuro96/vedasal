import _init_paths  # noqa: F401
from pre_processing.gen_points_anns import convert_xy_with_border


def test_convert_xy_with_border():
    # [(x, y), in_hw, out_hw
    conds = [
        [(0, 0), (768, 1024), (216, 384)],
        [(512, 768), (768, 1024), (216, 384)],
        [(1024, 768), (768, 1024), (216, 384)],
        [(1920, 1080), (1080, 1920), (216, 384)],
        [(0, 0), (1080, 1920), (216, 384)],
        # badcase
        [(1080, 1920), (1080, 1920), (216, 384)],
    ]
    for cond in conds:
        old_xy = cond[0]
        new_xy = convert_xy_with_border(*cond)
        print(f'cond {cond}: {old_xy} vs {new_xy}')


if __name__ == '__main__':
    test_convert_xy_with_border()
