from pyoof import aperture, telgeometry, fit_beam, extract_data_effelsberg

# telescope = [blockage, delta, pr, name]
telescope = dict(
    effelsberg=[
        telgeometry.block_effelsberg,
        telgeometry.delta_effelsberg,
        50,  # primary refelctor radius
        'effelsberg'
        ],
    manual=[
        telgeometry.block_manual(pr=50, sr=3.25, a=0, L=0),
        telgeometry.delta_effelsberg,
        50,  # primary refelctor radius
        'effelsberg partial blockage'
        ]
    )

illumination = dict(
    gaussian=[aperture.illum_gauss, 'gaussian', 'sigma_dB'],
    pedestal=[aperture.illum_pedestal, 'pedestal', 'c_dB']
    )


def fit_beam_effelsberg(pathfits):

    data_info, data_obs = extract_data_effelsberg(pathfits)

    [name, pthto, freq, wavel, d_z, meanel] = data_info
    [beam_data, u_data, v_data] = data_obs

    fit_beam(
        data_info=[name, pthto, freq, wavel, d_z, meanel],
        data_obs=[beam_data, u_data, v_data],
        order_max=6,  # it'll fit from 1 to order_max
        illumination=illumination['pedestal'],
        telescope=telescope['effelsberg'],
        fit_previous=True,
        angle='degrees',  # or radians
        resolution=2**8,  # standard is 2 ** 8
        make_plots=True
        )


if __name__ == '__main__':

    import glob  # to list as a string files in a directory
    # Directory for the fits files
    observation = glob.glob('../../data/S9mm_bump/*.fits')[0]
    fit_beam_effelsberg(observation)  # Execute!
