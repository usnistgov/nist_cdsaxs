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
    
    for y in range(0, grid_points_yx[0]):
        for x in range(0, grid_points_yx[1]):
            positions.append(((y+0.5)*y_space + center_yx[0] - region_of_interest_yx[0]/2, (x+0.5)*x_space + center_yx[1] - region_of_interest_yx[1]/2))

    assert x_space > beam_size_yx[1], "points are too close for this beam size along x-axis"
    assert y_space > beam_size_yx[0], "points are too close for this beam size along y-axis"

    return positions


def write_to_log_file(log_filename, sample_filename, sample_phi_deg, exposure_time_s):
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
    #print(save_data)
    data_dir="/nsls2/data/smi/legacy/results/analysis/2025_3/315554_Kline/"
    #data_dir="/nsls2/data/smi/proposals/2025-3/pass-315554/projects/cd-saxs/user_data/2M/"
    # we will check if this is a new log file and add a header
    # otherwise we will add the new row of metadata into the log
    with open(data_dir+log_filename, "a+") as file:
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


def measure_single_position(theta, exp_t=1, sample='test', nume=1, det=[pil2M], log_filepath="log_default_SMI_Kline_Nov2025.csv"):

    det_exposure_time(exp_t, exp_t*nume)

    sdd_cm = pil2M.sample_distance_mm.get()/10
    name_fmt = "{sample}_sdd_cm_{sdd_cm}_energy_ev_16100_sample_phi_deg_{th}_exposure_time_s_{et}_bpm_{bpm}_posx_um_{posx}_posy_um_{posy}_posz_um_{posz}"

    sample_name = name_fmt.format(
        sample=sample, 
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
        log_filename=log_filepath,
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
            log_filename=log_filepath,
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

                    print('moving to x, y', xs+rel_x, ys+rel_y)

                    # make sure that the y motor actually reaches position
                    while abs(piezo.y.position - (ys + rel_y)) >= 1:
                        print('y-motor did not reach position; requesting again')
                        yield from bps.mv(piezo.y, ys + rel_y)
                        yield from bps.sleep(5)

                    yield from measure_single_position(
                        phi_offset, exp_t=t, sample=f'{name}_grid'+'%s'%(ii+1), nume=repeats,
                        log_filepath = 'log_RR1_KlineNov25.csv'
                    )

def cdsaxs_Nov2025_grid_scans_CaitlynRoundRobin2(t=1):
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

    # from the front bottom left start naming
    # from the back start aligning from top right and move to bottom right
    names = ['RR80B',  'RR50A',  'RR23A',  'RR50H',  'RR23H',  'RR80H',  'RR80A',  'RR50B',  'RR23B']
    x =     [    670,   -12430,   -25430,   -38530,   -44030,   -30830,   -17330,    -4430,    8570]
    y =     [  -6800,    -6300,    -6100,    -6000,     6600,     6600,     6600,     6200,    6200]
    z =     [ 900-50,  1700-50,  1500-50,  1300-50,  1300-50,  1300-50,  1200-50,   700-50,   700-50]   

    names.reverse()
    x.reverse()
    y.reverse()
    z.reverse()
 

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
        for nn, (name, xs, ys, zs) in enumerate(zip(names, x, y, z)):

            if nn>=start_at:
                print(name)
                print('center positions x, y, z', xs, ys, zs)
                yield from bps.mv(piezo.x, xs)
                yield from bps.mv(piezo.y, ys)
                yield from bps.mv(piezo.z, zs)

                # make sure that the y motor actually reaches position
                while abs(piezo.y.position - ys) >= 1:
                    print('y-motor did not reach position; requesting again')
                    yield from bps.mv(piezo.y, ys)
                    yield from bps.sleep(5)

                for ii, (rel_y, rel_x) in enumerate(rel_positions):

                    yield from bps.mv(piezo.x, xs+rel_x)
                    yield from bps.mv(piezo.y, ys+rel_y)

                    print('moving to x, y', xs+rel_x, ys+rel_y)

                    # make sure that the y motor actually reaches position
                    while abs(piezo.y.position - (ys + rel_y)) >= 1:
                        print('y-motor did not reach position; requesting again')
                        yield from bps.mv(piezo.y, ys + rel_y)
                        yield from bps.sleep(5)

                    yield from measure_single_position(
                        phi_offset, exp_t=t, sample=f'{name}_grid'+'%s'%(ii+1), nume=repeats,
                        log_filepath = 'log_RR2_KlineNov25.csv'
                    )

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

    scans = list(['x', 'y'])#, 'z', 'phi']), #'chi', 'th', 'phi']  # make list of one or more motors in this list: ['x', 'y', 'z', 'chi', 'th', 'phi']
    scan_ranges = {
        # for each motor give a (start, stop, step_size)
        # or give a list of values
        # any motors listed here but not in 'scans' won't be used
        'x': (-300, 300, 50), #Values used during run were -0.3, 0.3, 0.05
        'y': (-300, 300, 50), #Values used during run were -0.3, 0.3, 0.05
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
    th =    [4,               3.2,               0,          0,       0,       0]

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
                yield from bps.mv(prs, phi_offset)
                yield from bps.mv(piezo.ch, chis)
                yield from bps.mv(piezo.th, ths)
                yield from bps.mv(piezo.z, zs)
                yield from bps.mv(piezo.x, xs)
                yield from bps.mv(piezo.y, ys)

                while abs(piezo.y.position - ys) >= 1:
                    print('y-motor did not reach position; requesting again')
                    yield from bps.mv(piezo.y, ys)
                    yield from bps.sleep(5)
            
                print(scans)    
                for sm, scan_motor in enumerate(scans[0]):
                    print(f'====== SCANNING {scan_motor} =======')
                    print(f'moving to phi {phi_offset}, chi {chis}, th {ths}, z {zs}, x {xs}, y {ys}')
                    yield from bps.mv(prs, phi_offset)
                    yield from bps.mv(piezo.ch, chis)
                    yield from bps.mv(piezo.th, ths)
                    yield from bps.mv(piezo.z, zs)
                    yield from bps.mv(piezo.x, xs)
                    yield from bps.mv(piezo.y, ys)

                    while abs(piezo.y.position - ys) >= 1:
                        print('y-motor did not reach position; requesting again')
                        yield from bps.mv(piezo.y, ys)
                        yield from bps.sleep(5)

                    positions = scan_ranges[scan_motor]
                    for ii, position in enumerate(positions):
                        if scan_motor == 'x':
                            print('moving x', position+xs)
                            yield from bps.mv(piezo.x, position + xs)
                        elif scan_motor == 'y':
                            print('moving y', position+ys)
                            yield from bps.mv(piezo.y, position + ys)
                            while abs(piezo.y.position - (position + ys)) >= 1:
                                print('y-motor did not reach position; requesting again')
                                yield from bps.mv(piezo.y, position + ys)
                                yield from bps.sleep(5)
                        elif scan_motor == 'z':
                            print('moving z', position+zs)
                            yield from bps.mv(piezo.z, position + zs)
                        elif scan_motor == 'chi':
                            print('moving chi', position+chis)
                            yield from bps.mv(piezo.ch, position + chis)
                        elif scan_motor == 'th':
                            print('moving th', position+ths)
                            yield from bps.mv(piezo.th, position + ths)
                        elif scan_motor == 'phi':
                            print('moving prs', position+phi_offset)
                            yield from bps.mv(prs, position + phi_offset)
                        else:
                            print("!!!!!!!! DIDN'T RECOGNIZE SCAN MOTOR !!!!!!!!")
                            continue
                        
                        theta_pass = phi_offset if scan_motor != 'phi' else phi_offset+position
        
                        yield from measure_single_position(
                            theta_pass, exp_t=t, sample=f'{name}_{scan_motor}-scan'+'%s'%(ii+1), nume=repeats, log_filepath='log_roundrobin1_misalignments_KlineNov25.csv'
                        )

def cdsaxs_Nov2025_misalignment_rescan_CaitlynRoundRobin1():
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

    scans = list(['x', 'y'])#, 'z', 'phi']), #'chi', 'th', 'phi']  # make list of one or more motors in this list: ['x', 'y', 'z', 'chi', 'th', 'phi']
    scan_ranges = {
        # for each motor give a (start, stop, step_size)
        # or give a list of values
        # any motors listed here but not in 'scans' won't be used
        'x': (-300, 300, 50),
        'y': (-300, 300, 50),
        'z': [-20000, -10000,  -9000,  -8000,  -7000,  -6000,  -5000,  -4000,  -3000,
            -2000,  -1000,   -500, -400, -300, -200, -100,    0,  100,  200,  300,  400,  500,
                1000,   2000,   3000,   4000,   5000,
            6000,   7000,   8000,   9000, 10000, 20000],
        'chi': (-1, 1, 0.1),
        'th': (-1, 1, 0.1),
        'phi': (-1, 1, 0.1),
    }

    scan_ranges_AgBeh = {
        # for each motor give a (start, stop, step_size)
        # or give a list of values
        # any motors listed here but not in 'scans' won't be used
        'x': (-1500, 1500, 250),
        'y': (-1000, 1000, 200),
        'z': [-20000, -10000,  -9000,  -8000,  -7000,  -6000,  -5000,  -4000,  -3000,
            -2000,  -1000,   -500, -400, -300, -200, -100,    0,  100,  200,  300,  400,  500,
                1000,   2000,   3000,   4000,   5000,
            6000,   7000,   8000,   9000, 10000, 20000],
        'chi': (-1, 1, 0.1),
        'th': (-1, 1, 0.1),
        'phi': (-1, 1, 0.1),
    }

    names = ['SRM_W204_F2', 'SRM_W204_H11', 'AgBeh']
    x =     [       -22670,         -22870,   47800]
    y=      [         7920,           2700,   -7100]
    z=      [          733,            733,     533]
    chi=    [         -1.3,             -3,       0]
    th =    [            4,            3.2,       0]

    

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

    ### reformat scan_ranges dictionary for easy scans
    for key, value in scan_ranges_AgBeh.items():
        if isinstance(value, tuple):
            new_positions = list(np.arange(value[0], value[1]+value[2]/10, value[2]))
            scan_ranges_AgBeh[key] = new_positions
    
    print(scan_ranges)
    print(scan_ranges_AgBeh)
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
                yield from bps.mv(prs, phi_offset)
                yield from bps.mv(piezo.ch, chis)
                yield from bps.mv(piezo.th, ths)
                yield from bps.mv(piezo.z, zs)
                yield from bps.mv(piezo.x, xs)
                yield from bps.mv(piezo.y, ys)

                while abs(piezo.y.position - ys) >= 1:
                    print('y-motor did not reach position; requesting again')
                    yield from bps.mv(piezo.y, ys)
                    yield from bps.sleep(5)
            
                print(scans)    
                for sm, scan_motor in enumerate(scans):
                    print(f'====== SCANNING {scan_motor} =======')
                    print(f'moving to phi {phi_offset}, chi {chis}, th {ths}, z {zs}, x {xs}, y {ys}')
                    yield from bps.mv(prs, phi_offset)
                    yield from bps.mv(piezo.ch, chis)
                    yield from bps.mv(piezo.th, ths)
                    yield from bps.mv(piezo.z, zs)
                    yield from bps.mv(piezo.x, xs)
                    yield from bps.mv(piezo.y, ys)

                    while abs(piezo.y.position - ys) >= 1:
                        print('y-motor did not reach position; requesting again')
                        yield from bps.mv(piezo.y, ys)
                        yield from bps.sleep(5)

                    if 'AgBeh' in name:
                        positions = scan_ranges_AgBeh[scan_motor]
                    else:
                        positions = scan_ranges[scan_motor]
                    for ii, position in enumerate(positions):
                        if scan_motor == 'x':
                            print('moving x', position+xs)
                            yield from bps.mv(piezo.x, position + xs)
                        elif scan_motor == 'y':
                            print('moving y', position+ys)
                            yield from bps.mv(piezo.y, position + ys)
                            while abs(piezo.y.position - (position + ys)) >= 1:
                                print('y-motor did not reach position; requesting again')
                                yield from bps.mv(piezo.y, position + ys)
                                yield from bps.sleep(5)
                        elif scan_motor == 'z':
                            print('moving z', position+zs)
                            yield from bps.mv(piezo.z, position + zs)
                        elif scan_motor == 'chi':
                            print('moving chi', position+chis)
                            yield from bps.mv(piezo.ch, position + chis)
                        elif scan_motor == 'th':
                            print('moving th', position+ths)
                            yield from bps.mv(piezo.th, position + ths)
                        elif scan_motor == 'phi':
                            print('moving prs', position+phi_offset)
                            yield from bps.mv(prs, position + phi_offset)
                        else:
                            print("!!!!!!!! DIDN'T RECOGNIZE SCAN MOTOR !!!!!!!!")
                            continue
                        
                        theta_pass = phi_offset if scan_motor != 'phi' else phi_offset+position
        
                        yield from measure_single_position(
                            theta_pass, exp_t=t, sample=f'{name}_{scan_motor}-scan'+'%s'%(ii+1), nume=repeats, log_filepath='log_roundrobin1_misalignments_rescan_KlineNov25.csv'
                        )


def cdsaxs_Nov2025_misalignment_scan_dupont2():
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

    scans = list(['th'])#, 'z', 'phi']), #'chi', 'th', 'phi']  # make list of one or more motors in this list: ['x', 'y', 'z', 'chi', 'th', 'phi']
    scan_ranges = {
        # for each motor give a (start, stop, step_size)
        # or give a list of values
        # any motors listed here but not in 'scans' won't be used
        'x': (-300, 300, 50),
        'y': (-300, 300, 50),
        'z': [-20000, -10000,  -9000,  -8000,  -7000,  -6000,  -5000,  -4000,  -3000,
            -2000,  -1000,   -500, -400, -300, -200, -100,    0,  100,  200,  300,  400,  500,
                1000,   2000,   3000,   4000,   5000,
            6000,   7000,   8000,   9000, 10000, 20000],
        'chi': (-2, 2, 0.1),
        'th': (-2, 2, 0.1),
        'phi': (-1, 1, 0.1),
    }

    # positions from facing back
    #           top right       
    names = ['EUVA_BD0_BF0',]
    x =     [         44770,]
    y=      [         -4100,]
    z=      [          -880,]
    chi=    [          -1.6,]
    th =    [           2.3,]

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
                t=1
                print(f'====== SCANNING {name} WITH EXPOSURE TIME {t}=======')
                print(f'moving to phi {phi_offset}, chi {chis}, th {ths}, z {zs}, x {xs}, y {ys}')
                yield from bps.mv(prs, phi_offset)
                yield from bps.mv(piezo.ch, chis)
                yield from bps.mv(piezo.th, ths)
                yield from bps.mv(piezo.z, zs)
                yield from bps.mv(piezo.x, xs)
                yield from bps.mv(piezo.y, ys)

                while abs(piezo.y.position - ys) >= 1:
                    print('y-motor did not reach position; requesting again')
                    yield from bps.mv(piezo.y, ys)
                    yield from bps.sleep(5)
            
                print(scans)    
                for sm, scan_motor in enumerate(scans):
                    print(f'====== SCANNING {scan_motor} =======')
                    print(f'moving to phi {phi_offset}, chi {chis}, th {ths}, z {zs}, x {xs}, y {ys}')
                    yield from bps.mv(prs, phi_offset)
                    yield from bps.mv(piezo.ch, chis)
                    yield from bps.mv(piezo.th, ths)
                    yield from bps.mv(piezo.z, zs)
                    yield from bps.mv(piezo.x, xs)
                    yield from bps.mv(piezo.y, ys)

                    while abs(piezo.y.position - ys) >= 1:
                        print('y-motor did not reach position; requesting again')
                        yield from bps.mv(piezo.y, ys)
                        yield from bps.sleep(5)

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
                            yield from bps.mv(piezo.ch, position + chis)
                        elif scan_motor == 'th':
                            print('moving th', position+ths)
                            yield from bps.mv(prs, 39)
                            yield from bps.mv(piezo.th, position + ths)
                        elif scan_motor == 'phi':
                            print('moving prs', position+phi_offset)
                            # yield from bps.mv(prs, position + phi_offset)
                        else:
                            print("!!!!!!!! DIDN'T RECOGNIZE SCAN MOTOR !!!!!!!!")
                            continue
                    
                        
                        theta_pass = phi_offset if scan_motor != 'phi' else phi_offset+position
        
                        yield from measure_single_position(
                            theta_pass, exp_t=t, sample=f'{name}_{scan_motor}-scan'+'%s'%(ii+1), nume=repeats, log_filepath='log_dupont2_misalignments_KlineNov25.csv'
                        )

                    yield from bps.mv(prs, phi_offset)



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
        for nn, (name, xs, ys, zs, chis, ths) in enumerate(zip(names, x, y, z, chi, th)):

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

def cdsaxs_Nov2025_chicago(t=5):
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
    x =     [             -13450,                 1350,                   16650,                  27650]
    y=      [              -8500,                -8500,                   -8500,                  -8500]
    z=      [                533,                  280,                     180,                  120]
    chi=    [              -0.95,                -0.95,                   -0.95,                  -1.4]
    th =    [                1.2,                  1.2,                     1.2,                  1.6]

    assert len(names) == len(x), f"len of x ({len(x)}) is different from number of samples ({len(names)})"
    assert len(names) == len(y), f"len of y ({len(y)}) is different from number of samples ({len(names)})"
    assert len(names) == len(z), f"len of z ({len(z)}) is different from number of samples ({len(names)})"
    assert len(names) == len(chi), f"len of chi ({len(chi)}) is different from number of samples ({len(names)})"
    assert len(names) == len(th), f"len of th ({len(th)}) is different from number of samples ({len(names)})"

    for i in range(1):
        for nn, (name, xs, ys, zs, chis, ths) in enumerate(zip(names, x, y, z, chi, th)):

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
                yield from cd_saxs(phi_offset, phi_offset, 1, exp_t=t, sample=name+'_measure_ref-A%s'%(i+1), nume=1, log_filepath='log_chicago_KlineNov25.csv')
                yield from cd_saxs(start_phi+phi_offset, stop_phi+phi_offset, phi_steps, exp_t=t, sample=name+'_measure%s'%(i+1), nume=repeats, log_filepath='log_chicago_KlineNov25.csv')
                yield from cd_saxs(phi_offset, phi_offset, 1, exp_t=t, sample=name+'_measure_ref-B%s'%(i+1), nume=1, log_filepath='log_chicago_KlineNov25.csv')


def cdsaxs_Nov2025_dupont_1(t=10):
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

    names = ['EUVB_BD2_BF0', 'EUVB_BDm2_BF0', 'EUVB_BD0_BF2',  'EUVB_BD0_BF1', 'EUVB_BD0_BF0', 'EUVB_BD0_BFm1', 'EUVB_BD0_BFm2']
    x =     [        -40850,          -27950,         -12550,            2200,          19000,           37700,          52300 ]
    y=      [          6000,            5300,           5300,            5300,           5500,            4700,           4700 ]
    z=      [           720,             220,           -280,            -880,          -1480,           -2280,          -2780 ]
    chi=    [          -0.2,             2.5,           -1.2,            -2.2,           -1.5,            -4.5,           -3.5 ]
    th =    [           2.1,               1,              2,               1,            1.5,             1.8,            2.5 ]

    names = names + ['EUVD_BD2_BF0', 'EUVD_BDm2_BF0', 'EUVD_BD0_BF3',  'EUVD_BD0_BF1', 'EUVD_BD0_BF0', 'EUVD_BD0_BFm1', 'EUVD_BD0_BFm3']
    x =     x     + [        -40150,          -25900,         -10900,            3000,          18200,           33200,          46770 ]
    y=      y     + [         -4000,           -5000,          -4900,           -4900,          -4700,           -4600,          -4100 ]
    z=      z     + [           920,             320,             20,            -680,          -1380,           -1980,          -2580 ]
    chi=    chi   + [          -7.3,               1,           -0.5,            -1.9,           -1.9,            -1.9,           -2.5 ]
    th =    th    + [           3.5,               3,            3.2,             3.5,            2.5,             2.5,            2.0 ]  



    print(names)
    print(x)
    print(y)
    print(z)
    print(chi)
    print(th)
    assert len(names) == len(x), f"len of x ({len(x)}) is different from number of samples ({len(names)})"
    assert len(names) == len(y), f"len of y ({len(y)}) is different from number of samples ({len(names)})"
    assert len(names) == len(z), f"len of z ({len(z)}) is different from number of samples ({len(names)})"
    assert len(names) == len(chi), f"len of chi ({len(chi)}) is different from number of samples ({len(names)})"
    assert len(names) == len(th), f"len of th ({len(th)}) is different from number of samples ({len(names)})"

    for i in range(1):
        for nn, (name, xs, ys, zs, chis, ths) in enumerate(zip(names, x, y, z, chi, th)):

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
                
                log_filename_string='log_dupont1_KlineNov25.csv'
                #yield from bp
                yield from cd_saxs(phi_offset, phi_offset, 1, exp_t=t, sample=name+'_measure_ref-A%s'%(i+1), nume=1, log_filepath=log_filename_string)
                yield from cd_saxs(start_phi+phi_offset, stop_phi+phi_offset, phi_steps, exp_t=t, sample=name+'_measure%s'%(i+1), nume=repeats, log_filepath=log_filename_string)
                yield from cd_saxs(phi_offset, phi_offset, 1, exp_t=t, sample=name+'_measure_ref-B%s'%(i+1), nume=1, log_filepath=log_filename_string)

def cdsaxs_Nov2025_dupont_2(t=10):
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

    # positions from facing back
    #          bottom left     bottom right      top right     top left           
    names = ['EUVC_BD0_BFm2', 'EUVC_BD0_BF0', 'EUVA_BD0_BF0', 'EUVA_BD0_BFm2',]
    x =     [          32470,          48370,          44770,           32170,]
    y=      [           6000,           6000,          -4100,           -4100,]
    z=      [           -680,           -980,           -880,            -580,]
    chi=    [            0.7,           -0.6,           -1.6,              -2,]
    th =    [            1.3,            2.0,            2.3,             2.8,]

    print(names)
    print(x)
    print(y)
    print(z)
    print(chi)
    print(th)
    assert len(names) == len(x), f"len of x ({len(x)}) is different from number of samples ({len(names)})"
    assert len(names) == len(y), f"len of y ({len(y)}) is different from number of samples ({len(names)})"
    assert len(names) == len(z), f"len of z ({len(z)}) is different from number of samples ({len(names)})"
    assert len(names) == len(chi), f"len of chi ({len(chi)}) is different from number of samples ({len(names)})"
    assert len(names) == len(th), f"len of th ({len(th)}) is different from number of samples ({len(names)})"

    for i in range(1):
        for nn, (name, xs, ys, zs, chis, ths) in enumerate(zip(names, x, y, z, chi, th)):

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
                
                log_filename_string='log_dupont2_KlineNov25.csv'
                # yield from bp
                yield from cd_saxs(phi_offset, phi_offset, 1, exp_t=t, sample=name+'_measure_ref-A%s'%(i+1), nume=1, log_filepath=log_filename_string)
                yield from cd_saxs(start_phi+phi_offset, stop_phi+phi_offset, phi_steps, exp_t=t, sample=name+'_measure%s'%(i+1), nume=repeats, log_filepath=log_filename_string)
                yield from cd_saxs(phi_offset, phi_offset, 1, exp_t=t, sample=name+'_measure_ref-B%s'%(i+1), nume=1, log_filepath=log_filename_string)

