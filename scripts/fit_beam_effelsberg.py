from pyoof import aperture, telgeometry, fit_beam, extract_data_effelsberg

# telescope = [blockage, delta, pr, name]
telescope = dict(
    effelsberg=[
        telgeometry.blockage_effelsberg,
        telgeometry.delta_effelsberg,
        50,  # primary refelctor radius
        'effelsberg'
        ],
    manual=[
        telgeometry.blockage_manual(pr=50, sr=3.25, a=0, L=0),
        telgeometry.delta_effelsberg,
        50,  # primary refelctor radius
        'effelsberg partial blockage'
        ]
    )

illumination = dict(
    gaussian=[aperture.illumination_gauss, 'gaussian', 'sigma_dB'],
    pedestal=[aperture.illumination_pedestal, 'pedestal', 'c_dB']
    )


def fit_beam_effelsberg(pathfits):

    data_info, data_obs = extract_data_effelsberg(pathfits)
    # [name, pthto, freq, wavel, d_z_m, meanel], [beam_data, u_data, v_data]
    # u and v must be in radians, beam will be normalised

    fit_beam(
        data=[data_info, data_obs],
        order_max=5,  # it'll fit from 1 to 7
        illumination=illumination['pedestal'],
        telescope=telescope['effelsberg'],
        fit_previous=True,
        angle='degrees',  # or radians
        resolution=2**8  # standard used is 2 ** 10
        )


if __name__ == '__main__':

    import glob  # to list as a string files in a directory

    # Change to desired data location in your machine
    observation = glob.glob('../../data/S9mm/*.fits')[0]  # len = 8
    # for obs in observation:
    fit_beam_effelsberg(observation)
