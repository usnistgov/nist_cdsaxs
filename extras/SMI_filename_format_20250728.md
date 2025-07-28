# Filename formatting at the SMI beamline

The following code can be incorporated into the Python control scripts
to create filenames consistent with the metadata keywords and other
naming conventions used in this Python library.

## Instructions
1. **Change the base sample name format in the `cd_saxs` function.**
Within the `cd_saxs` function two lines should be replaced with the
following code:
```
name_fmt = "{sample}_sdd_cm_520_energy_ev_16100_sample_phi_deg_{th}_exposure_time_s_{et}_I0_{bpm}_num{num}"
sample_name = name_fmt.format(sample=sample, num="%4.4d"%num, th="%06.2f"%theta, et= "%.2f"%exp_t, bpm="%1.3f"%xbpm3.sumX.get())
```
**NOTE**: the energy and nominal sample-to-detector distance are
hardcoded in this format. This should be updated with the beamline
scientist during your experiment as required! We can always correct
these metadata values during reduction but ideally this would be
noted correctly from the start.

* `sample`: custom sample name provided by the user
during alignment and will be passed to this function. It is assumed
that this will not include an underscore at the end of it. The other
standard metadata saved from the configuration settings are:
* `num`: interative integer that captures the acquisition number of the
`cd_saxs` call. For example, this number would vary from 0 to 121 for a
sample phi scan from -60 to 60 with a step size of 1. Currently, the
understanding of C.Wolf is that this only applies to the sample phi as
it is the only loop built into the function. It is formatted as a signed
integer with a width of 4 digits that are filled with preceeding 0's. 
* `theta`: This is the sample rotation angle and what we refer to as
`sample_phi_deg` in our code. It is formatted as a float value with
two decimal places and a width of 6 including the decimal and a negative
sign if the value is less than 0.
* `ept_t`: exposure time of each image acquisition. This can be pulled
from the metadata of the tiff file from the Pilatus detector, but it is
also convenient to pull it from the filename and also as a visual
reminder for the researcher viewing the files. It is formatted as
a float with two decimal places.
* `bpm` is the SMI monitor and is left in the original formatting. It
is formatted as a float with 3 decimal places.

The name format is designed to include all the critical metadata
required for the basic reduction workflow in the `nist_cdsaxs`
library but other keywords can be added to that variable if required.
If the metadata is related to a keyword that is already defined in
src/cdsaxs/metadata.py, the keyword can be included in the filename
exactly as it is written and the value for that meatadata should
immediately follow the keyword, separated with an additional underscore.
For examle, adding the count time involved adding "exposure_time_s_{et}"
to the `name_fmt` variable then formatting with the `exp_t` variable in
the control script.

2. **Change the call to the `cd_saxs` function in each of the
user-defined cdsaxs functions.** An example of this function from
the last beamtime in March 2025 was `def cdsaxs_2025_1_Matt(t=5): ...`.
A portion of the original version looked something like this after the
sample positions were all defined:
```
for i in range(1):
        for nn, (name, xs, ys, zs, chis, ths) in enumerate(zip(names, x, y, z, chi, th)):
            # yield from bps.mv(stage.x, xs_hexa)
            # yield from bps.mv(stage.y, ys_hexa)
            
            if nn>=0:
                yield from bps.mv(piezo.z, zs)
                yield from bps.mv(piezo.ch, chis)
                yield from bps.mv(piezo.th, ths)
                yield from bps.mv(piezo.x, xs)
                yield from bps.mv(piezo.y, ys)
                
                # force piezo.y to move to the correct position
                while abs(piezo.y.position - ys) >= 1:
                    print('y motor error')
                    yield from bps.mv(piezo.y, ys)
                    yield from bps.sleep(4)
                
                number = 1              
        

                # yield from bp
                yield from cd_saxs(phi_offest, phi_offest, 1, exp_t=t, sample=name+'measure_ref-A%s'%(i+1), nume=1)
                yield from cd_saxs(-46+phi_offest, 44+phi_offest, 46, exp_t=t, sample=name+'measure1%s'%(i+1), nume=number)
                yield from cd_saxs(phi_offest, phi_offest, 1, exp_t=t, sample=name+'measure_ref-B%s'%(i+1), nume=1)
                yield from cd_saxs(-45+phi_offest, 45+phi_offest, 46, exp_t=t, sample=name+'measure2%s'%(i+1), nume=number)
                yield from cd_saxs(phi_offest, phi_offest, 1, exp_t=t, sample=name+'measure_ref-C%s'%(i+1), nume=1)
                yield from cd_saxs(-46+phi_offest, 44+phi_offest, 46, exp_t=t, sample=name+'measure3%s'%(i+1), nume=number)
                yield from cd_saxs(phi_offest, phi_offest, 1, exp_t=t, sample=name+'measure_ref-D%s'%(i+1), nume=1)
                yield from cd_saxs(-45+phi_offest, 45+phi_offest, 46, exp_t=t, sample=name+'measure4%s'%(i+1), nume=number)
                yield from cd_saxs(phi_offest, phi_offest, 1, exp_t=t, sample=name+'measure_ref-E%s'%(i+1), nume=1)
```

The last few lines that start with `yield from cdsaxs(...)` should be
modified so that there is an underscore between `name` and `measure...`.
This will make filtering filenames based on the same name easier! Also,
please consider the use of the 'ref' and (i+1) integers from the for
loop and how best to organize your files. The `name_fmt` that we
defined above will follow whatever you put here and is the `sample`
keyword we reference above.

## Quick formatting guide for styling numbers in the filename string
The numbers can be formatted using the modulo operator, %. It follows
the format:

"%(width).(precision)(keyword)" % value
* width: minimum length of the resulting string   
    * formatting the integer 4 with width of 2 would result in " 4"
    where there is an extra space between the digit of 4.
    * formatting the integer 10000 with a width of 2 would still result
    in "10000". 
* precision: number of digits behind the decimal place
    * formatting the number 1.2335 as a float with 3 decimal places
    would result in the string "1.234".
* keyword: defines the format of the number. Common choices here will
be 'f' for a floating point and 'd' for signed integer.
* value: the number that is to be formated as a string using the
previous modifiers

Examples:
* `"%4.d" % 5.5` results in <code>'&nbsp;&nbsp;&nbsp;5'</code>.
* `"%4.d" % 12345` results in `'12334'`.
* `"%04.d" % 5.5` results in `'0005'`. Use the 0 to include preceeding 0's
when you need to fill in the extra spaces.
* `"%.1f" % 8.326502` results in `'8.3'`
* `"%.3f" % 8.326502` results in `'8.327'`
* `"%6.3f" % 8.326502` results in `' 8.327'`.
