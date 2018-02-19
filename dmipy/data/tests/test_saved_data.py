from dmipy.data.saved_data import (
    duval_cat_spinal_cord_2d,
    duval_cat_spinal_cord_3d
)


def test_loading_saved_duval_data_2dand3d():
    scheme, data = duval_cat_spinal_cord_2d()
    scheme, data = duval_cat_spinal_cord_3d()
