def gen_grid_points_yx(region_of_interest_yx, center_yx, beam_size_yx, grid_points_yx):

    """
    NOTE: all parameters should be provided in the same units. If your motor position for center_yx is based in um, then your region_of_interest_yx and beam_size_yx should also be based in um.

    region_of_interest_yx : tuple
        Available measurement area.
    center_yx : tuple
        Center motor position.
    beam_size_yx : tuple
        Dimensions of the beam.
    grid_points_yx : tuple
        Number of points along each axis in the grid.
        (2, 4) will result in a 2 x 4 grid of points, 8 measurement points total.
    """

    x_space = region_of_interest_yx[1]/(grid_points_yx[1])
    y_space = region_of_interest_yx[0]/(grid_points_yx[0])
    print('x_space', x_space, 'y_space', y_space)

    positions = []
    positions.append(center_yx)
    for x in range(0, grid_points_yx[1]):
        for y in range(0, grid_points_yx[0]):
            positions.append(((y+0.5)*y_space + center_yx[0] - region_of_interest_yx[0]/2, (x+0.5)*x_space + center_yx[1] - region_of_interest_yx[1]/2))

    assert x_space > beam_size_yx[1], "points are too close for this beam size along x-axis"
    assert y_space > beam_size_yx[0], "points are too close for this beam size along y-axis"

    return positions


def write_to_log_file(log_filepath, sample_filename, sample_phi_deg, exposure_time_s):
    # get all current motor positions
    # add sample_phi_deg motor position if we can figure that out during beamtime
    sdd_cm = pil2M.sample_distance_mm.get()/10
    sample_chi_deg = piezo.ch.position
    sample_omega_deg = piezo.th.position
    bpm = xbpm3.sumX.get()
    posx_um = piezo.x.position
    posy_um = piezo.y.position
    posz_um = piezo.z.position

    # make a list of all metadata for the given sample filename
    save_data = [
        sample_filename,
        exposure_time_s,
        sdd_cm,
        sample_phi_deg,
        sample_chi_deg,
        sample_omega_deg,
        bpm,
        posx_um,
        posy_um,
        posz_um,
    ]

    # we will check if this is a new log file and add a header
    # otherwise we will add the new row of metadata into the log
    with open(log_filepath, "a+") as file:
        file.seek(0)
        current_data = file.readlines()
        if len(current_data) == 0:
            file.write("sample_filename, exposure_time_s, sdd_cm, sample_phi_deg, sample_chi_deg, sample_omega_deg, bpm, posx_um, posy_um, posz_um\n")
        file.write(', '.join(str(item) for item in save_data)+"\n")

 
def measure(det=[pil2M], sample='test',  t=1):
    det_exposure_time(t, t)
    sample_name = "{sample}".format(sample=sample)
    sample_id(user_name="JK", sample_name=sample_name)
    print(f"\n\t=== Sample: {sample_name} ===\n")
    yield from bp.count(det, num=1)


def measure_single_position(theta, exp_t=1, sample='test', nume=1, det=[pil2M], log_filepath="./log_SMI_Kline_Nov2025.csv"):

    det_exposure_time(exp_t, exp_t*nume)

    sdd_cm = pil2M.sample_distance_mm.get()/10
    name_fmt = "{sample}_sdd_cm_{sdd_cm}_energy_ev_16100_sample_phi_deg_{th}_exposure_time_s_{et}_bpm_{bpm}_posx_um_{posx}_posy_um_{posy}_posz_um_{posz}_num_{num}"

    sample_name = name_fmt.format(
        sample=sample, 
        num="%2.2d"%num,
        th="%2.2d"%theta,
        bpm="%1.3f"%xbpm3.sumX.get(),
        et = "%2.2f"%exp_t,
        sdd_cm = "%.2f"%sdd_cm,
        posx = "%.0f"%piezo.x.position,
        posy = "%.0f"%piezo.y.position,
        posz = "%.0f"%piezo.z.position)
    # sample_id(user_name="JK", sample_name=sample_name)
    sample_id(sample_name=sample_name)
    print(f"\n\t=== Sample: {sample_name} ===\n")

    # if we can figure out how to extract the actual full filename
    # including the id then we should use that in the log file instead
    write_to_log_file(
        log_filepath=log_filepath,
        sample_filename=sample_name,
        sample_phi_deg=theta,
        exposure_time_s=exp_t
    )
    yield from bp.count(det, num=1)


def cd_saxs(th_ini, th_fin, th_st, exp_t=1, sample='test', nume=1, det=[pil2M], log_filepath="./log_SMI_Kline_Nov2025.csv"):

    det_exposure_time(exp_t, exp_t*nume)

    for num, theta in enumerate(np.linspace(th_ini, th_fin, th_st)):
        yield from bps.mv(prs, theta)
        sdd_cm = pil2M.sample_distance_mm.get()/10
        name_fmt = "{sample}_sdd_cm_{sdd_cm}_energy_ev_16100_sample_phi_deg_{th}_exposure_time_s_{et}_bpm_{bpm}_posx_um_{posx}_posy_um_{posy}_posz_um_{posz}_num_{num}"

        sample_name = name_fmt.format(
            sample=sample, 
            num="%2.2d"%num, 
            th="%2.2d"%theta,
            bpm="%1.3f"%xbpm3.sumX.get(),
            et = "%2.2f"%exp_t,
            sdd_cm = "%.2f"%sdd_cm,
            posx = "%.0f"%piezo.x.position,
            posy = "%.0f"%piezo.y.position,
            posz = "%.0f"%piezo.z.position)
        # sample_id(user_name="JK", sample_name=sample_name)
        sample_id(sample_name=sample_name)
        print(f"\n\t=== Sample: {sample_name} ===\n")

        # if we can figure out how to extract the actual full filename
        # including the id then we should use that in the log file instead
        write_to_log_file(
            log_filepath=log_filepath,
            sample_filename=sample_name,
            sample_phi_deg=theta,
            exposure_time_s=exp_t
        )
        yield from bp.count(det, num=1)

def cdsaxs_Nov2025_template(t=1):
    """
    If you need to restart this sample set at a sample other than the
    first one, change the 'start_at' variable. 
    Samples are indexed starting at 0, so if 'start_at' is equal to 0,
    all samples in this set will be run with this function call.

    The repeats parameter is used to collect multiple
    images each with an expsoure time of t at each position during
    the cd-saxs scan.
    """
    det = [pil2M]
    
    phi_offset = 0
    start_phi = -60
    stop_phi = 60
    phi_steps = int(abs(start_phi-stop_phi) + 1)

    start_at = 0
    repeats = 1

    names = [ 'sampleA', 'sampleB', 'sampleC']
    x =     [   -350, 550, 1450]
    y=      [    3450, 3450, 3450]
    z=      [    -4600, -4600, -4600]
    chi=    [    -1.6, -1.6, -1.6]
    th =    [  3.5, 3.5, 3.5]

    assert len(names) == len(x), f"len of x ({len(x)}) is different from number of samples ({len(names)})"
    assert len(names) == len(y), f"len of y ({len(y)}) is different from number of samples ({len(names)})"
    assert len(names) == len(z), f"len of z ({len(z)}) is different from number of samples ({len(names)})"
    assert len(names) == len(chi), f"len of chi ({len(chi)}) is different from number of samples ({len(names)})"
    assert len(names) == len(th), f"len of th ({len(th)}) is different from number of samples ({len(names)})"

    for i in range(1):
        for nn, (name, xs, ys, zs, chis, ths) in enumerate(zip(names, x, y, z)):

            if nn>=start_at:
                yield from bps.mv(piezo.ch, chis)
                yield from bps.mv(piezo.th, ths)
                yield from bps.mv(piezo.z, zs)
                yield from bps.mv(piezo.x, xs)
                yield from bps.mv(piezo.y, ys)

                # make sure that the y motor actually reaches position
                while abs(piezo.y.position - ys) >= 1:
                    print('y-motor did not reach position; requesting again')
                    yield from bps.mv(piezo.y, ys)
                    yield from bps.sleep(5)
                
                # number = 1              
            
                # yield from bp
                yield from cd_saxs(phi_offset, phi_offset, 1, exp_t=t, sample=name+'_measure_ref-A%s'%(i+1), nume=1)
                yield from cd_saxs(start_phi+phi_offset, stop_phi+phi_offset, phi_steps, exp_t=t, sample=name+'_measure%s'%(i+1), nume=repeats)
                yield from cd_saxs(phi_offset, phi_offset, 1, exp_t=t, sample=name+'_measure_ref-B%s'%(i+1), nume=1)

def cdsaxs_Nov2025_template_odds_evens(t=1):
    """
    If you need to restart this sample set at a sample other than the
    first one, change the 'start_at' variable. 
    Samples are indexed starting at 0, so if 'start_at' is equal to 0,
    all samples in this set will be run with this function call.

    The repeats parameter is used to collect multiple
    images each with an expsoure time of t at each position during
    the cd-saxs scan.
    """
    
    det = [pil2M]
    
    phi_offset = 0
    min_phi = -45 # pick an integer value
    max_phi = 45 # pick an integer value

    start_at = 0
    repeats = 1

    min_even = min_phi if min_phi%2 == 0 else min_phi + 1
    max_even = max_phi if max_phi%2 == 0 else max_phi - 1
    min_odd = min_phi if min_phi%2 != 0 else min_phi + 1
    max_odd = max_phi if max_phi%2 != 0 else max_phi - 1


    names = [ 'sampleA', 'sampleB', 'sampleC']
    x =     [   -350, 550, 1450]
    y=      [    3450, 3450, 3450]
    z=      [    -4600, -4600, -4600]
    chi=    [    -1.6, -1.6, -1.6]
    th =    [  3.5, 3.5, 3.5]

    assert len(names) == len(x), f"len of x ({len(x)}) is different from number of samples ({len(names)})"
    assert len(names) == len(y), f"len of y ({len(y)}) is different from number of samples ({len(names)})"
    assert len(names) == len(z), f"len of z ({len(z)}) is different from number of samples ({len(names)})"
    assert len(names) == len(chi), f"len of chi ({len(chi)}) is different from number of samples ({len(names)})"
    assert len(names) == len(th), f"len of th ({len(th)}) is different from number of samples ({len(names)})"
    
    for i in range(1):
        for nn, (name, xs, ys, zs, chis, ths) in enumerate(zip(names, x, y, z)):

            if nn>=start_at:
                yield from bps.mv(piezo.ch, chis)
                yield from bps.mv(piezo.th, ths)
                yield from bps.mv(piezo.z, zs)
                yield from bps.mv(piezo.x, xs)
                yield from bps.mv(piezo.y, ys)

                # make sure that the y motor actually reaches position
                while abs(piezo.y.position - ys) >= 1:
                    print('y-motor did not reach position; requesting again')
                    yield from bps.mv(piezo.y, ys)
                    yield from bps.sleep(5)
                
                # number = 1              
            
                yield from cd_saxs(phi_offset, phi_offset, 1, exp_t=t, sample=name+'measure_ref-A%s'%(i+1), nume=1)
                yield from cd_saxs(min_even+phi_offset, max_even+phi_offset, int(((max_even - min_even)/2)+1), exp_t=t, sample=name+'measure1%s'%(i+1), nume=repeats)
                yield from cd_saxs(phi_offset, phi_offset, 1, exp_t=t, sample=name+'measure_ref-B%s'%(i+1), nume=1)
                yield from cd_saxs(min_odd+phi_offset, max_odd+phi_offset, int(((max_odd - min_odd)/2)+1), exp_t=t, sample=name+'measure2%s'%(i+1), nume=repeats)
                yield from cd_saxs(phi_offset, phi_offset, 1, exp_t=t, sample=name+'measure_ref-C%s'%(i+1), nume=1)

def cdsaxs_Nov2025_template_grid_scans(t=1):
    """
    If you need to restart this sample set at a sample other than the
    first one, change the 'start_at' variable. 
    Samples are indexed starting at 0, so if 'start_at' is equal to 0,
    all samples in this set will be run with this function call.

    The repeats parameter is used to collect multiple
    images each with an expsoure time of t at each position during
    the cd-saxs scan.
    """
    det = [pil2M]
    
    phi_offset = 0
    start_phi = -60
    stop_phi = 60
    phi_steps = int(abs(start_phi-stop_phi) + 1)

    start_at = 0
    repeats = 1

    # x, y, z should be the center position for each sample
    names = [ 'sampleA', 'sampleB', 'sampleC']
    x =     [   -350, 550, 1450]
    y=      [    3450, 3450, 3450]
    z=      [    -4600, -4600, -4600]
    chi=    [    -1.6, -1.6, -1.6]
    th =    [  3.5, 3.5, 3.5]

    range_x = 4500 # um, or same units as x, y, z
    range_y = 3000 # um, or same units as x, y, z

    rel_positions = gen_grid_points_yx(
        region_of_interest_yx=(range_y, range_x),
        center_yx=(0, 0),
        beam_size_yx=(250, 25),
        grid_points_yx=(4, 2)
    )

    print(f"========= Measuring at relative grid positions: {rel_positions}.")

    assert len(names) == len(x), f"len of x ({len(x)}) is different from number of samples ({len(names)})"
    assert len(names) == len(y), f"len of y ({len(y)}) is different from number of samples ({len(names)})"
    assert len(names) == len(z), f"len of z ({len(z)}) is different from number of samples ({len(names)})"
    assert len(names) == len(chi), f"len of chi ({len(chi)}) is different from number of samples ({len(names)})"
    assert len(names) == len(th), f"len of th ({len(th)}) is different from number of samples ({len(names)})"
    
    for i in range(1):
        for nn, (name, xs, ys, zs, chis, ths) in enumerate(zip(names, x, y, z)):

            if nn>=start_at:
                yield from bps.mv(piezo.ch, chis)
                yield from bps.mv(piezo.th, ths)
                yield from bps.mv(piezo.z, zs)
                yield from bps.mv(piezo.x, xs)
                yield from bps.mv(piezo.y, ys)

                # make sure that the y motor actually reaches position
                while abs(piezo.y.position - ys) >= 1:
                    print('y-motor did not reach position; requesting again')
                    yield from bps.mv(piezo.y, ys)
                    yield from bps.sleep(5)
                
                for ii, (rel_y, rel_x) in enumerate(rel_positions):

                    yield from bps.mv(piezo.x, xs+rel_x)
                    yield from bps.mv(piezo.y, ys+rel_y)

                    # make sure that the y motor actually reaches position
                    while abs(piezo.y.position - ys + rel_y) >= 1:
                        print('y-motor did not reach position; requesting again')
                        yield from bps.mv(piezo.y, ys + rel_y)
                        yield from bps.sleep(5)

                    yield from measure_single_position(
                        exp_t=t, sample=f'grid-scan'+'%s'%(ii+1), nume=repeats
                    )  

def cdsaxs_Nov2025_template_motor_scan(t=1):
    """
    If you need to restart this sample set at a sample other than the
    first one, change the 'start_at' variable. 
    Samples are indexed starting at 0, so if 'start_at' is equal to 0,
    all samples in this set will be run with this function call.

    The repeats parameter is used to collect multiple
    images each with an expsoure time of t at each position during
    the cd-saxs scan.

    """
    det = [pil2M]

    phi_offset = 0
    # start_phi = -60
    # stop_phi = 60
    # this will give steps every 1 degree
    # phi_steps = int(abs(start_phi-stop_phi) + 1)

    start_at = 0
    repeats = 1

    scans = None # make list of one or more motors in this list: ['x', 'y', 'z', 'chi', 'th']
    scan_ranges = {
        # for each motor give a (start, stop, step_size)
        # or give a list of values
        # any motors listed here but not in 'scans' won't be used
        'x': (-0.3, 0.3, 0.05),
        'y': (-0.3, 0.3, 0.05),
        'z': [-20000, -10000,  -9000,  -8000,  -7000,  -6000,  -5000,  -4000,  -3000,
            -2000,  -1000,   -500, -400, -300, -200, -100,    0,  100,  200,  300,  400,  500,
                1000,   2000,   3000,   4000,   5000,
            6000,   7000,   8000,   9000, 10000, 20000],
        'chi': (-1, 1, 0.1),
        'th': (-1, 1, 0.1),
    }

    names = [ 'sampleA', 'sampleB', 'sampleC']
    x =     [   -350, 550, 1450]
    y=      [    3450, 3450, 3450]
    z=      [    -4600, -4600, -4600]
    chi=    [    -1.6, -1.6, -1.6]
    th =    [  3.5, 3.5, 3.5]

    assert len(names) == len(x), f"len of x ({len(x)}) is different from number of samples ({len(names)})"
    assert len(names) == len(y), f"len of y ({len(y)}) is different from number of samples ({len(names)})"
    assert len(names) == len(z), f"len of z ({len(z)}) is different from number of samples ({len(names)})"
    assert len(names) == len(chi), f"len of chi ({len(chi)}) is different from number of samples ({len(names)})"
    assert len(names) == len(th), f"len of th ({len(th)}) is different from number of samples ({len(names)})"
    

    ### reformat scan_ranges dictionary for easy scans
    for key, value in scan_ranges.items():
        if isinstance(value, tuple):
            new_positions = list(np.arange(value[0], value[1]+value[2]/10, value[2]))
            scan_ranges[key] = new_positions
    
    if isinstance(positions, tuple):
                    positions = list(np.arange(positions[0], positions[1]+positions[2]/10, positions[2]))
    for i in range(1):
        for nn, (name, xs, ys, zs, chis, ths) in enumerate(zip(names, x, y, z)):

            if nn>=start_at:
                yield from bps.mv(prs, phi_offset)
                yield from bps.mv(piezo.ch, chis)
                yield from bps.mv(piezo.th, ths)
                yield from bps.mv(piezo.z, zs)
                yield from bps.mv(piezo.x, xs)
                yield from bps.mv(piezo.y, ys)

                # make sure that the y motor actually reaches position
                while abs(piezo.y.position - ys) >= 1:
                    print('y-motor did not reach position; requesting again')
                    yield from bps.mv(piezo.y, ys)
                    yield from bps.sleep(5)
                # number = 1              
            
                # # yield from bp
                # yield from cd_saxs(phi_offset, phi_offset, 1, exp_t=t, sample=name+'_measure_ref-A%s'%(i+1), nume=1)
                # yield from cd_saxs(start_phi+phi_offset, stop_phi+phi_offset, phi_steps, exp_t=t, sample=name+'_measure%s'%(i+1), nume=repeats)
                # yield from cd_saxs(phi_offset, phi_offset, 1, exp_t=t, sample=name+'_measure_ref-B%s'%(i+1), nume=1)

                for scan_motor in scans:
                    print (f'====== SCANNING {scan_motor} =======')
                    yield from bps.mv(prs, phi_offset)
                    yield from bps.mv(piezo.ch, chis)
                    yield from bps.mv(piezo.th, ths)
                    yield from bps.mv(piezo.z, zs)
                    yield from bps.mv(piezo.x, xs)
                    yield from bps.mv(piezo.y, ys)

                    # make sure that the y motor actually reaches position
                    while abs(piezo.y.position - ys) >= 1:
                        print('y-motor did not reach position; requesting again')
                        yield from bps.mv(piezo.y, ys)
                        yield from bps.sleep(5)

                    positions = scan_ranges[scan_motor]
                    for ii, position in enumerate(positions):
                        if scan_motor == 'x':
                            yield from bps.mv(piezo.x, position + xs)
                        elif scan_motor == 'y':
                            yield from bps.mv(piezo.y, position + ys)
                            while abs(piezo.y.position - (position + ys)) >= 1:
                                print('y-motor did not reach position; requesting again')
                                yield from bps.mv(piezo.y, position + ys)
                                yield from bps.sleep(5)
                        elif scan_motor == 'z':
                            yield from bps.mv(piezo.z, position + zs)
                        elif scan_motor == 'chi':
                            yield from bps.mv(piezo.ch, position + chis)
                        elif scan_motor == 'th':
                            yield from bps.mv(piezo.th, position + ths)
                        elif scan_motor == 'phi':
                            yield from bps.mv(prs, position + phi_offset)
                        else:
                            print("!!!!!!!! DIDN'T RECOGNIZE SCAN MOTOR !!!!!!!!")
                            continue

                        yield from measure_single_position(
                            exp_t=t, sample=f'{scan_motor}-scan'+'%s'%(ii+1), nume=repeats
                        )


def cdsaxs_Nov2025_grid_scans_CaitlynRoundRobin1(t=10):
    """
    If you need to restart this sample set at a sample other than the
    first one, change the 'start_at' variable.
    Samples are indexed starting at 0, so if 'start_at' is equal to 0,
    all samples in this set will be run with this function call.

    The repeats parameter is used to collect multiple
    images each with an expsoure time of t at each position during
    the cd-saxs scan.
    """
    det = [pil2M]

    phi_offset = -6
    # start_phi = -60
    # stop_phi = 60
    # phi_steps = int(abs(start_phi-stop_phi) + 1)

    start_at = 0
    repeats = 1

    # x, y, z should be the center position for each sample
    # names = [ 'RR50C', 'RR23C', 'RR80E', 'RR50G', 'RR23D', 'RR80D', 'RR50F', 'RRAgBeh', 'RR80C', 'RR50E', 'SRM_W204_H11', 'SRM_W204_F2', 'RR23G', 'RR80G', 'RR50D', 'RR23F', 'RR80F']
    names = ['RR80C',  'RR50E', 'RR23G','RR80G', 'RR50D', 'RR23F', 'RR80F', 'RR50F', 'RR80D', 'RR23D', 'RR50G', 'RR80E', 'RR23C', 'RR50C']
    x =     [   -45270, -32270, -11470, 1530,    14530,     27330,  40329,   33330 ,  20131,     6631,   -6369,  -19670,  -32670, -45469]
    y=      [     7020,   7020,   7320, 7620,     7620,     7620,   7820,    -7480 ,  -7480,    -7480,   -7480,  -7680,   -7880,   -7880]


    range_x = 4500 # um, or same units as x, y, z
    range_y = 3000 # um, or same units as x, y, z

    rel_positions = gen_grid_points_yx(
        region_of_interest_yx=(range_y, range_x),
        center_yx=(0, 0),
        beam_size_yx=(25, 250),
        grid_points_yx=(2, 4)
    )

    print(f"========= Measuring at relative grid positions (y, x): {rel_positions}.")

    assert len(names) == len(x), f"len of x ({len(x)}) is different from number of samples ({len(names)})"
    assert len(names) == len(y), f"len of y ({len(y)}) is different from number of samples ({len(names)})"

    for i in range(1):
        for nn, (name, xs, ys) in enumerate(zip(names, x, y)):

            if nn>=start_at:
                print(name)
                print('center positions x, y', xs, ys)
                # yield from bps.mv(piezo.x, xs)
                # yield from bps.mv(piezo.y, ys)

                # make sure that the y motor actually reaches position
                # while abs(piezo.y.position - ys) >= 1:
                #     print('y-motor did not reach position; requesting again')
                #     yield from bps.mv(piezo.y, ys)
                #     yield from bps.sleep(5)

                for ii, (rel_y, rel_x) in enumerate(rel_positions):

                    # yield from bps.mv(piezo.x, xs+rel_x)
                    # yield from bps.mv(piezo.y, ys+rel_y)

                    print('moving to x, y', xs+rel_x, ys+rel_y)

                    # make sure that the y motor actually reaches position
                    # while abs(piezo.y.position - ys + rel_y) >= 1:
                    #     print('y-motor did not reach position; requesting again')
                    #     yield from bps.mv(piezo.y, ys + rel_y)
                    #     yield from bps.sleep(5)

                    # yield from measure_single_position(
                    #     phi_offset, exp_t=t, sample='name'+f'_grid'+'%s'%(ii+1), nume=repeats
                    # )


def cdsaxs_Nov2025_misalignment_scan_CaitlynRoundRobin1():
    """
    If you need to restart this sample set at a sample other than the
    first one, change the 'start_at' variable. 
    Samples are indexed starting at 0, so if 'start_at' is equal to 0,
    all samples in this set will be run with this function call.

    The repeats parameter is used to collect multiple
    images each with an expsoure time of t at each position during
    the cd-saxs scan.

    """
    det = [pil2M]

    phi_offset = -6

    start_at = 0
    repeats = 1

    scans = ['x', 'y', 'z', 'phi'], #'chi', 'th', 'phi']  # make list of one or more motors in this list: ['x', 'y', 'z', 'chi', 'th', 'phi']
    scan_ranges = {
        # for each motor give a (start, stop, step_size)
        # or give a list of values
        # any motors listed here but not in 'scans' won't be used
        'x': (-0.3, 0.3, 0.05),
        'y': (-0.3, 0.3, 0.05),
        'z': [-20000, -10000,  -9000,  -8000,  -7000,  -6000,  -5000,  -4000,  -3000,
            -2000,  -1000,   -500, -400, -300, -200, -100,    0,  100,  200,  300,  400,  500,
                1000,   2000,   3000,   4000,   5000,
            6000,   7000,   8000,   9000, 10000, 20000],
        'chi': (-1, 1, 0.1),
        'th': (-1, 1, 0.1),
        'phi': (-1, 1, 0.1),
    }

    names = ['SRM_W204_F2', 'SRM_W204_H11', 'RR50D',  'RR23F', 'RR80F', 'AgBeh']# 'RR50F', 'RR80D', 'RR23D', 'RR50G', 'RR80E', 'RR23C', 'RR50C']
    x =     [-22470,        -22570,          14530,     27330,  40329,   48000] #, 33330 ,  20131,     6631,   -6369,  -19670,  -32670, -45469]
    y=      [7820,          2600,             7620,     7620,   7820,    -7280] #, -7480 ,  -7480,    -7480,   -7480,  -7680,   -7880,   -7880]
    z=      [533,          633,                533,      533,      533,    533]
    chi=    [-1.3,            -3,               0,          0,       0,       0]
    th =    [4,              3.2,               0,          0,       0,       0]

    assert len(names) == len(x), f"len of x ({len(x)}) is different from number of samples ({len(names)})"
    assert len(names) == len(y), f"len of y ({len(y)}) is different from number of samples ({len(names)})"
    assert len(names) == len(z), f"len of z ({len(z)}) is different from number of samples ({len(names)})"
    assert len(names) == len(chi), f"len of chi ({len(chi)}) is different from number of samples ({len(names)})"
    assert len(names) == len(th), f"len of th ({len(th)}) is different from number of samples ({len(names)})"
    
    ### reformat scan_ranges dictionary for easy scans
    for key, value in scan_ranges.items():
        if isinstance(value, tuple):
            new_positions = list(np.arange(value[0], value[1]+value[2]/10, value[2]))
            scan_ranges[key] = new_positions
    
    print(scan_ranges)
    for i in range(1):
        for nn, (name, xs, ys, zs, chis, ths) in enumerate(zip(names, x, y, z, chi, th)):

            if nn>=start_at:
                if 'RR' in name:
                    t = 1
                elif 'Ag' in name:
                    t = 0.5
                else:
                    t = 0.1
                print(f'====== SCANNING {name} WITH EXPOSURE TIME {t}=======')
                print(f'moving to phi {phi_offset}, chi {chis}, th {ths}, z {zs}, x {xs}, y {ys}')
                # yield from bps.mv(prs, phi_offset)
                # yield from bps.mv(piezo.ch, chis)
                # yield from bps.mv(piezo.th, ths)
                # yield from bps.mv(piezo.z, zs)
                # yield from bps.mv(piezo.x, xs)
                # yield from bps.mv(piezo.y, ys)

                # while abs(piezo.y.position - ys) >= 1:
                #     print('y-motor did not reach position; requesting again')
                #     yield from bps.mv(piezo.y, ys)
                #     yield from bps.sleep(10)
            
            
                for scan_motor in scans:
                    print(f'====== SCANNING {scan_motor} =======')
                    print(f'moving to phi {phi_offset}, chi {chis}, th {ths}, z {zs}, x {xs}, y {ys}')
                    # yield from bps.mv(prs, phi_offset)
                    # yield from bps.mv(piezo.ch, chis)
                    # yield from bps.mv(piezo.th, ths)
                    # yield from bps.mv(piezo.z, zs)
                    # yield from bps.mv(piezo.x, xs)
                    # yield from bps.mv(piezo.y, ys)

                    # while abs(piezo.y.position - ys) >= 1:
                    #     print('y-motor did not reach position; requesting again')
                    #     yield from bps.mv(piezo.y, ys)
                    #     yield from bps.sleep(10)

                    positions = scan_ranges[scan_motor]
                    for ii, position in enumerate(positions):
                        if scan_motor == 'x':
                            print('moving x', position+xs)
                            # yield from bps.mv(piezo.x, position + xs)
                        elif scan_motor == 'y':
                            print('moving y', position+ys)
                            # yield from bps.mv(piezo.y, position + ys)
                            # while abs(piezo.y.position - (position + ys)) >= 1:
                            #     print('y-motor did not reach position; requesting again')
                            #     yield from bps.mv(piezo.y, position + ys)
                            #     yield from bps.sleep(5)
                        elif scan_motor == 'z':
                            print('moving z', position+zs)
                            # yield from bps.mv(piezo.z, position + zs)
                        elif scan_motor == 'chi':
                            print('moving chi', position+chis)
                            # yield from bps.mv(piezo.ch, position + chis)
                        elif scan_motor == 'th':
                            print('moving th', position+ths)
                            # yield from bps.mv(piezo.th, position + ths)
                        elif scan_motor == 'phi':
                            print('moving prs', position+phi_offset)
                            # yield from bps.mv(prs, position + phi_offset)
                        else:
                            print("!!!!!!!! DIDN'T RECOGNIZE SCAN MOTOR !!!!!!!!")
                            continue

        
                        # yield from measure_single_position(
                        #     exp_t=t, sample=f'{name}_{scan_motor}-scan'+'%s'%(ii+1), nume=repeats, log_filename='log_roundrobin1_misalignments_KlineNov25.csv'
                        # ) 

def cdsaxs_Nov2025_itri(t=0.5):
    """
    If you need to restart this sample set at a sample other than the
    first one, change the 'start_at' variable. 
    Samples are indexed starting at 0, so if 'start_at' is equal to 0,
    all samples in this set will be run with this function call.

    The repeats parameter is used to collect multiple
    images each with an expsoure time of t at each position during
    the cd-saxs scan.
    """
    det = [pil2M]
    
    phi_offset = -6
    start_phi = -60
    stop_phi = 60
    phi_steps = int(abs(start_phi-stop_phi) + 1)

    start_at = 0
    repeats = 1

    names = ['itri_nsh30', 'itri_nsh00', 'itri_nsh10', 'itri_nsh20' ]#, 'chicago_Set1_SM26', 'chicago_Set1_ABC26', 'chicago_Set2_0.8L0_24', 'chicago_Set2_1.25L0_54']
    x =     [      -48900,        46100,        43750,       -36450 ]#,             'xxxx',               'xxxx',                   'xxxx',                  'xxxx']
    y=      [        7150,         6000,        -8380,       -10000 ]#,             'xxxx',               'xxxx',                   'xxxx',                  'xxxx']
    z=      [         833,          133,         -167,          533 ]#,             'xxxx',               'xxxx',                   'xxxx',                  'xxxx']
    chi=    [         0.9,         -1.2,         -1.8,        -0.95 ]#,             'xxxx',               'xxxx',                   'xxxx',                  'xxxx']
    th =    [           2,            2,          1.2,          1.2 ]#,             'xxxx',               'xxxx',                   'xxxx',                  'xxxx']

    assert len(names) == len(x), f"len of x ({len(x)}) is different from number of samples ({len(names)})"
    assert len(names) == len(y), f"len of y ({len(y)}) is different from number of samples ({len(names)})"
    assert len(names) == len(z), f"len of z ({len(z)}) is different from number of samples ({len(names)})"
    assert len(names) == len(chi), f"len of chi ({len(chi)}) is different from number of samples ({len(names)})"
    assert len(names) == len(th), f"len of th ({len(th)}) is different from number of samples ({len(names)})"

    for i in range(1):
        for nn, (name, xs, ys, zs, chis, ths) in enumerate(zip(names, x, y, z)):

            if nn>=start_at:
                print(f'====== SCANNING {name} WITH EXPOSURE TIME {t}=======')
                print(f'moving to phi {phi_offset}, chi {chis}, th {ths}, z {zs}, x {xs}, y {ys}')
                yield from bps.mv(piezo.ch, chis)
                yield from bps.mv(piezo.th, ths)
                yield from bps.mv(piezo.z, zs)
                yield from bps.mv(piezo.x, xs)
                yield from bps.mv(piezo.y, ys)

                # make sure that the y motor actually reaches position
                while abs(piezo.y.position - ys) >= 1:
                    print('y-motor did not reach position; requesting again')
                    yield from bps.mv(piezo.y, ys)
                    yield from bps.sleep(5)
                
            
                # yield from bp
                yield from cd_saxs(phi_offset, phi_offset, 1, exp_t=t, sample=name+'_measure_ref-A%s'%(i+1), nume=1, log_filepath='log_itri_chicago_KlineNov25.csv')
                yield from cd_saxs(start_phi+phi_offset, stop_phi+phi_offset, phi_steps, exp_t=t, sample=name+'_measure%s'%(i+1), nume=repeats, log_filepath='log_itri_chicago_KlineNov25.csv')
                yield from cd_saxs(phi_offset, phi_offset, 1, exp_t=t, sample=name+'_measure_ref-B%s'%(i+1), nume=1, log_filepath='log_itri_chicago_KlineNov25.csv')

def cdsaxs_Nov2025_chicago(t=1):
    """
    If you need to restart this sample set at a sample other than the
    first one, change the 'start_at' variable. 
    Samples are indexed starting at 0, so if 'start_at' is equal to 0,
    all samples in this set will be run with this function call.

    The repeats parameter is used to collect multiple
    images each with an expsoure time of t at each position during
    the cd-saxs scan.
    """
    det = [pil2M]
    
    phi_offset = -6
    start_phi = -60
    stop_phi = 60
    phi_steps = int(abs(start_phi-stop_phi) + 1)

    start_at = 0
    repeats = 1

    names = ['chicago_Set1_SM26', 'chicago_Set1_ABC26', 'chicago_Set2_0.8L0_24', 'chicago_Set2_1.25L0_54']
    x =     [             'xxxx',               'xxxx',                   'xxxx',                  'xxxx']
    y=      [             'xxxx',               'xxxx',                   'xxxx',                  'xxxx']
    z=      [             'xxxx',               'xxxx',                   'xxxx',                  'xxxx']
    chi=    [             'xxxx',               'xxxx',                   'xxxx',                  'xxxx']
    th =    [             'xxxx',               'xxxx',                   'xxxx',                  'xxxx']

    assert len(names) == len(x), f"len of x ({len(x)}) is different from number of samples ({len(names)})"
    assert len(names) == len(y), f"len of y ({len(y)}) is different from number of samples ({len(names)})"
    assert len(names) == len(z), f"len of z ({len(z)}) is different from number of samples ({len(names)})"
    assert len(names) == len(chi), f"len of chi ({len(chi)}) is different from number of samples ({len(names)})"
    assert len(names) == len(th), f"len of th ({len(th)}) is different from number of samples ({len(names)})"

    for i in range(1):
        for nn, (name, xs, ys, zs, chis, ths) in enumerate(zip(names, x, y, z)):

            if nn>=start_at:
                print(f'====== SCANNING {name} WITH EXPOSURE TIME {t}=======')
                print(f'moving to phi {phi_offset}, chi {chis}, th {ths}, z {zs}, x {xs}, y {ys}')
                yield from bps.mv(piezo.ch, chis)
                yield from bps.mv(piezo.th, ths)
                yield from bps.mv(piezo.z, zs)
                yield from bps.mv(piezo.x, xs)
                yield from bps.mv(piezo.y, ys)

                # make sure that the y motor actually reaches position
                while abs(piezo.y.position - ys) >= 1:
                    print('y-motor did not reach position; requesting again')
                    yield from bps.mv(piezo.y, ys)
                    yield from bps.sleep(5)
                
            
                # yield from bp
                yield from cd_saxs(phi_offset, phi_offset, 1, exp_t=t, sample=name+'_measure_ref-A%s'%(i+1), nume=1, log_filepath='log_itri_chicago_KlineNov25.csv')
                yield from cd_saxs(start_phi+phi_offset, stop_phi+phi_offset, phi_steps, exp_t=t, sample=name+'_measure%s'%(i+1), nume=repeats, log_filepath='log_itri_chicago_KlineNov25.csv')
                yield from cd_saxs(phi_offset, phi_offset, 1, exp_t=t, sample=name+'_measure_ref-B%s'%(i+1), nume=1, log_filepath='log_itri_chicago_KlineNov25.csv')
