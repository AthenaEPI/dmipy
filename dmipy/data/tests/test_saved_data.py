from dmipy.data.saved_data import (
    duval_cat_spinal_cord_2d,
    duval_cat_spinal_cord_3d,
    isbi2015_white_matter_challenge,
    panagiotaki_verdict
)


def test_loading_saved_duval_data_2dand3d():
    scheme, data = duval_cat_spinal_cord_2d()
    scheme, data = duval_cat_spinal_cord_3d()
    scheme, data1, data2 = isbi2015_white_matter_challenge()
    scheme, data = panagiotaki_verdict()
