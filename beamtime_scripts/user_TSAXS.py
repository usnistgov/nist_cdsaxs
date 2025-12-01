#!/usr/bin/python
# -*- coding: utf-8 -*-
# vi: ts=4 sw=4




################################################################################
#  Short-term settings (specific to a particular user/experiment) can
# be placed in this file. You may instead wish to make a copy of this file in
# the user's data directory, and use that as a working copy.
################################################################################


#logbooks_default = ['User Experiments']
#tags_default = ['CFN Soft-Bio']

import pickle
import os
from shutil import copyfile

from ophyd import EpicsSignal
from bluesky.suspenders import SuspendFloor, SuspendCeil

ring_current = EpicsSignal('SR:OPS-BI{DCCT:1}I:Real-I')
# if True:
#     sus = SuspendFloor(ring_current, 100, resume_thresh=400, sleep=600)
#     RE.install_suspender(sus)

if True:
    sus = SuspendFloor(ring_current, 100, resume_thresh=350, sleep=600)
    RE.install_suspender(sus)

# Set experiment directories and calibration
RE.md['experiment_alias_directory'] = '0_TSAXS'
RE.md["userpy_alias_directory"] = '/nsls2/data/cms/shared/config/bluesky/profile_collection/users/2025-3/JKline'
cms.SAXS.setCalibration([746, 1080], 5.03, [-65, -73])  #2024 June 5m

### DEFINE YOUR PARENT DATA FOLDER HERE 


if False:
    # The following shortcuts can be used for unit conversions. For instance,
    # for a motor operating in 'mm' units, one could instead do:
    #     sam.xr( 10*um )
    # To move it by 10 micrometers. HOWEVER, one must be careful if using
    # these conversion parameters, since they make implicit assumptions.
    # For instance, they assume linear axes are all using 'mm' units. Conversely,
    # you will not receive an error if you try to use 'um' for a rotation axis!
    m = 1e3
    cm = 10.0
    mm = 1.0
    um = 1e-3
    nm = 1e-6
    
    inch = 25.4
    pixel = 0.172 # Pilatus
    
    deg = 1.0
    rad = np.degrees(1.0)
    mrad = np.degrees(1e-3)
    urad = np.degrees(1e-6)


def get_default_stage():
    return stg


class SampleTSAXS(SampleTSAXS_Generic):
    
    def __init__(self, name, base=None, **md):
        super().__init__(name=name, base=base, **md)
        self.naming_scheme = ['name', 'extra', 'exposure_time']

class Sample(SampleTSAXS):
# class Sample(SampleGISAXS):

    def __init__(self, name, base=None, **md):
       
       super().__init__(name=name, base=base, **md)

       self.naming_scheme = ['name', 'extra', 'x', 'y', 'exposure_time']

       self._axes['y'].origin = 10
       self._axes['th'].origin = 0
       
       self.md['exposure_time'] = 1 
       self.SAXS_time = 5
       self.WAXS_time = 20
       
       self.incident_angles_default = [0.0]

       self.x_pos_default = [-1, 0, 1]

    
    def goto(self, label, verbosity=3, **additional):
        super().goto(label, verbosity=verbosity, **additional)
        # You can add customized 'goto' behavior here
        
    def scan_SAXSdet(self, exposure_time=None) :
        SAXS_pos=[-73, 0, 73]
        #SAXSx_pos=[-65, 0, 65]
        
        RE.md['stitchback'] = True
                
        for SAXSx_pos in SAXS_pos:
            for SAXSy_pos in SAXS_pos:
                mov(SAXSx, SAXSx_pos)
                mov(SAXSy, SAXSy_pos)
                self.measure(10)
            
        
    def IC_int(self):
        
        ion_chamber_readout1=caget('XF:11BMB-BI{IM:3}:IC1_MON')
        ion_chamber_readout2=caget('XF:11BMB-BI{IM:3}:IC2_MON')
        ion_chamber_readout3=caget('XF:11BMB-BI{IM:3}:IC3_MON')
        ion_chamber_readout4=caget('XF:11BMB-BI{IM:3}:IC4_MON')
        
        ion_chamber_readout=ion_chamber_readout1+ion_chamber_readout2+ion_chamber_readout3+ion_chamber_readout4
        
        return ion_chamber_readout>1*5e-08
    
    def intMeasure(self, output_file, exposure_time=1):
        '''Measure the transmission intensity of the sample by ROI4.
        The intensity will be saved in output_file
        '''        
        if abs(beam.energy(verbosity=0) - 13.5) < 0.1:
            #beam.setAbsorber(4)
            beam.setTransmission(1e-4)
        elif abs(beam.energy(verbosity=0) - 17) < 0.1:
            beam.setTransmission(1e-6)

        # print('Absorber is moved to position {}'.format(beam.absorber()[0]))

        detselect([pilatus2M])
        #if beam.absorber()[0]>=4:
        bsx.move(bsx.position+6)
            #beam.setTransmission(1)
            
        self.measure(exposure_time)
        
        temp_data = self.transmission_data_output(4)

        cms.modeMeasurement()
        #beam.setAbsorber(0)
        #beam.absorber_out()
       
        #output_data = output_data.iloc[0:0]

        #create a data file to save the INT data
        INT_FILENAME='{}/data/{}.csv'.format(os.path.dirname(__file__) , output_file)            
        
        if os.path.isfile(INT_FILENAME):
            output_data = pds.read_csv(INT_FILENAME, index_col=0)
            output_data = pds.concat([output_data, temp_data])    
            output_data.to_csv(INT_FILENAME)
        else:
            temp_data.to_csv(INT_FILENAME)

    def transmission_data_output(self, slot_pos):
        '''Output the tranmission of direct beam
        '''
        h = db[-1]
        dtable = h.table()
        
        #beam.absorber_transmission_list = [1, 0.041, 0.0017425, 0.00007301075, 0.00000287662355, 0.000000122831826, 0.00000000513437]
        scan_id = h.start['scan_id']     
        I_bim5 = h.start['beam_int_bim5']  #beam intensity from bim5
        I0 = dtable.pilatus2M_stats4_total
        filename = h.start['sample_name']
        exposure_time = h.start['sample_exposure_time']
        #I2 = dtable.pilatus2M_stats2_total
        #I3 = 2*dtable.pilatus2M_stats1_total - dtable.pilatus2M_stats2_total
        #In = I3 / beam.absorber_transmission_list[slot_pos] / exposure_time

        current_data = {'a_filename': filename,
                        'b_scanID': scan_id,
                        'c_I0': I0,
                        'd_I_bim5': I_bim5,
                        'e_absorber_slot': slot_pos,
                        #'f_absorber_ratio': beam.absorber_transmission_list[slot_pos],
                        'f_absorber_ratio': 0.000001,
                        'g_exposure_seconds': exposure_time}
        
        return pds.DataFrame(data=current_data)   
            

class CapillaryHolderThreeRowsCustom(CapillaryHolderThreeRows):

    def __init__(self, name='CapillaryHolderThreeRowsCustom', base=None, **kwargs):
        super().__init__(name=name, base=base, **kwargs)

        #QCAPHolder
        self._axes['y'].origin = -4.2
        self._axes['x'].origin = -16.55795 #as of 2025C3

        self.WAXS_time=10

    #def doSamples(self, verbosity=3):

        ##maxs_on()
        #for sample in self.getSamples():
            #if verbosity>=3:
                #print('Doing sample {}...'.format(sample.name))
            #if sample.detector=='SAXS' or sample.detector=='BOTH':
                #sample.do_SAXS()

        #for sample in self.getSamples():
            #if verbosity>=3:
                #print('Doing sample {}...'.format(sample.name))
            #if sample.detector=='BOTH':
                #sample.do_WAXS_only()

        #for sample in self.getSamples():
            #if verbosity>=3:
                #print('Doing sample {}...'.format(sample.name))
            #if sample.detector=='WAXS':
                #sample.do_WAXS()
 
    def intMeasures(self,output_file='Transmission_output'):
        for sample in self.getSamples():
            sample.gotoOrigin()
            while smx.moving==True:
                time.sleep(0.2)
            sample.intMeasure(output_file=output_file)  

    # def doSamples(self, sequence='Outer',  exposure_WAXS_time=10, exposure_SAXS_time=60, verbosity=3):
        
    #     if sequence =='Outer':
    #         step = 0
    #     elif sequence == 'Inner':
    #         step = 5
    #     else: 
    #         return print('Please define the first measurement: Outer or Inner')
    #     if step< 1:
            
    #         waxs_on_outer()
    #         for sample in self.getSamples():
    #             sample.gotoOrigin()
    #             while smx.moving:
    #                 time.sleep(0.3)
    #             sample.measure(exposure_WAXS_time, extra='outer-normal', tiling='ygaps') 

    #         detselect(pilatus2M)
    #         for sample in self.getSamples():
    #             sample.gotoOrigin()
    #             while smx.moving:
    #                 time.sleep(0.3)
    #             sample.measure(exposure_SAXS_time, tiling='ygaps') 

    #     if step< 10:
    #         waxs_on_inner()
    #         for sample in self.getSamples():
    #             sample.gotoOrigin()
    #             while smx.moving:
    #                 time.sleep(0.3)
    #             sample.measure(exposure_WAXS_time, extra='inner-normal', tiling='ygaps') 


    #     if step > 1:
    #         waxs_on_outer()
    #         for sample in self.getSamples():
    #             sample.gotoOrigin()
    #             while smx.moving:
    #                 time.sleep(0.3)
    #             sample.measure(exposure_WAXS_time, extra='outer-normal', tiling='ygaps') 

    #         detselect(pilatus2M)
    #         for sample in self.getSamples():
    #             sample.gotoOrigin()
    #             while smx.moving:
    #                 time.sleep(0.3)
    #             sample.measure(exposure_SAXS_time, tiling='ygaps') 


    def doSamples_custom(self, exposure_WAXS_time=10):

        swaxs_on()
        for sample in self.getSamples():
            sample.gotoOrigin()
            sample.measure(exposure_WAXS_time, tiling='ygaps') 
            # sample.measure(exposure_WAXS_time, tiling=tiling) 
        # for sample in self.getSamples():
        #     sample.gotoOrigin()
        #     sample.intMeasure(output_file='Transmission_output') 
        


    
def swaxs_on():
    detselect([pilatus2M, pilatus800])
    WAXSx.move(-192)        
    WAXSy.move(16)    


def saxs_on():
    detselect(pilatus2M)
    #WAXSx.move(-195)        
    #WAXSy.move(24)    

def saxs_on_det():
    detselect(pilatus2M)
    WAXSx.move(-200)        
    WAXSy.move(30)    

def waxs_on():
    detselect(pilatus800)
    WAXSx.move(-195)    
    WAXSy.move(18)   

def waxs_on_outer():   #for inner-outer stitching
    # detselect(pilatus800)
    detselect([pilatus2M,pilatus800])
    WAXSx.move(-240)
    WAXSy.move(50)

def waxs_on_inner():   #for inner-outer stitching
    detselect(pilatus800)
    # WAXSx.move(-210)
    # WAXSy.move(20)
    WAXSx.move(-195)    
    WAXSy.move(17)  





if True:
    
    cali = CapillaryHolder(base=stg)
    #hol = CapillaryHolderCustom(base=stg)
    
    cali.addSampleSlot( Sample('Lab6_cali_5m_13.5kev'), 2.0 )
    cali.addSampleSlot( Sample('FL_screen'), 5.0 )
    cali.addSampleSlot( Sample('AgBH_cali_5m_13.5kev'), 7.0 )
    cali.addSampleSlot( Sample('AgBHCeO2_cali_5m_13.5kev'), 8.0 )
    cali.addSampleSlot( Sample('Empty'), 11.0 )

if True:
    
    # Example of a multi-sample holder
    
    md = {
        'owner' : 'J.Kline (NIST)' ,
        'series' : 'various' ,
        }


    hol1 = CapillaryHolderThreeRowsCustom(base=stg)
    hol1.addGaragePosition(1,1)
    hol1.name = 'hol1'


    hol1.addSampleSlot( Sample('RR50C'),1)
    hol1.addSampleSlot( Sample('RR23C'),2)
    hol1.addSampleSlot( Sample('RR80E'),3)
    hol1.addSampleSlot( Sample('RR50G'),4)
    hol1.addSampleSlot( Sample('RR23D'),5)
    hol1.addSampleSlot( Sample('RR80D'),6)
    hol1.addSampleSlot( Sample('RR50F'),7)
    hol1.addSampleSlot( Sample('RR80F'),8) #snaking over samples NOT rastering
    hol1.addSampleSlot( Sample('RR23F'),9)
    hol1.addSampleSlot( Sample('RR50D'),10)
    hol1.addSampleSlot( Sample('RR80G'),11)
    hol1.addSampleSlot( Sample('RR23E'),12)
    hol1.addSampleSlot( Sample('RR50E'),13)
    hol1.addSampleSlot( Sample('RR80C'),14)


    hol2 = CapillaryHolderThreeRowsCustom(base=stg)
    hol2.addGaragePosition(1,2)
    hol2.name = 'hol2'
    hol2.addSampleSlot( Sample('AS_1A'),1)
    hol2.addSampleSlot( Sample('AS_1B'),2)
    hol2.addSampleSlot( Sample('AS_2A'),3)
    hol2.addSampleSlot( Sample('AS_2B'),4)
    hol2.addSampleSlot( Sample('AS_2C'),5)
    hol2.addSampleSlot( Sample('AS_2D'),6)
    hol2.addSampleSlot( Sample('AS_3A'),7)
    hol2.addSampleSlot( Sample('AS_3B'),8)
    hol2.addSampleSlot( Sample('AS_3C'),9)
    hol2.addSampleSlot( Sample('AS_3D'),10)
    hol2.addSampleSlot( Sample('AS_3E'),11)
    hol2.addSampleSlot( Sample('AS_4A'),12)
    hol2.addSampleSlot( Sample('AS_4B'),13)
    hol2.addSampleSlot( Sample('AS_4C'),14)
    hol2.addSampleSlot( Sample('AS_4D'),15)
    hol2.addSampleSlot( Sample('AS_4E'),16)
    hol2.addSampleSlot( Sample('AS_5A'),17)
    hol2.addSampleSlot( Sample('AS_5B'),18)
    hol2.addSampleSlot( Sample('AS_5C'),19)
    hol2.addSampleSlot( Sample('AS_5D'),20)
    hol2.addSampleSlot( Sample('AS_6A'),21)
    hol2.addSampleSlot( Sample('AS_6B'),22)
    hol2.addSampleSlot( Sample('AS_6C'),23)
    hol2.addSampleSlot( Sample('AS_6D'),24)
    hol2.addSampleSlot( Sample('AS_7A'),25)
    hol2.addSampleSlot( Sample('AS_7B'),26)
    hol2.addSampleSlot( Sample('AS_7C'),27)
    hol2.addSampleSlot( Sample('AS_7D'),28)
    hol2.addSampleSlot( Sample('AS_8A'),29)
    hol2.addSampleSlot( Sample('AS_8B'),30)
    hol2.addSampleSlot( Sample('AS_9A'),31)
    hol2.addSampleSlot( Sample('AS_9B'),32)
    hol2.addSampleSlot( Sample('AS_9C'),33)
    hol2.addSampleSlot( Sample('AS_9D'),34)
    hol2.addSampleSlot( Sample('AS_CNC'),35)
    hol2.addSampleSlot( Sample('AS_10A'),36)
    hol2.addSampleSlot( Sample('AS_10B'),37)
    hol2.addSampleSlot( Sample('AS_10C'),38)
    hol2.addSampleSlot( Sample('AS_11A'),39)
    hol2.addSampleSlot( Sample('AS_11B'),40)
    hol2.addSampleSlot( Sample('AS_11C'),41)
    hol2.addSampleSlot( Sample('AS_11D'),42)
    hol2.addSampleSlot( Sample('AS_11E'),43)
    hol2.addSampleSlot( Sample('AS_11F'),44)
    hol2.addSampleSlot( Sample('AS_12A'),45)





    # que = Queue(base=stg)
    # que.addHolderIntoQueue(hol1, [1, 1], 1) 
    # que.addHolderIntoQueue(hol2, [1, 2], 2)
    # que.addHolderIntoQueue(hol3, [1, 3], 3)  
    # que.addHolderIntoQueue(hol4, [2, 1], 4) 
    # que.addHolderIntoQueue(hol5, [2, 2], 5)
    # que.addHolderIntoQueue(hol6, [2, 3], 6)
    # que.addHolderIntoQueue(hol7, [3, 1], 7)
    # # # que.addHolderIntoQueue(hol1, [2, 2], 4) 
    # # # #que.addHolderIntoQueue(hol5, [2, 2], 5) 
    # # # #que.addHolderIntoQueue(hol6, [2, 3], 6) 
    # # # ##que.addHolderIntoQueue(hol7, [3, 1], 7) 
    # # # #que.addHolderIntoQueue(hol8, [3, 2], 8) 
    # # # #que.addHolderIntoQueue(hol9, [3, 3], 9) 
    # # # #que.addHolderIntoQueue(hol10, [4, 1], 10) 
    # # # #que.addHolderIntoQueue(hol11, [1, 2], 2) 
    # # # #que.addHolderIntoQueue(hol10, [4, 1], 10) 
    # # # #que.addHolderIntoQueue(hol11, [4, 2], 11) 
    # # # #que.addHolder(hol3, [2, 3]) 
    # # # que.addHolderIntoQueue(cali, [3, 1], 10) 
    # que.setSequence()  
    # que.checkStatus(verbosity=5) 





# robot.listGarage()


'''
2023_3 (Sept)

#################################
Run 2

17 keV, TSAXS/WAXS, in vacuum
beam size (0.2, 0.2)

round beamstop
In [75]: wbs()
bsx = -18.393963
bsy = -0.2998839999999997
bsphi = -20.012715

ThreeRows
Capillary




#################################
Run 1

17 keV, edge on, in vacuum
beam size (0.2, 0.2)

rod beamstop
In [558]: wbs()
bsx = -15.591719999999999
bsy = 17.000437
bsphi = -177.99958500000002

offcenter sample holder, 
find the good sample spot manually and set origin 'x' and 'y'
measure stitched, inner and outer

'''










'''
New alignment procedure for holder:

1. move to the sample close to the center (cali_sample)
2. align the cali_sample and set the other samples position (y and th) as cali_sample
3. align the other samples one by one
   --1>> y scan in narrow range (0.6 in 21 steps) 
   --2>> 
   --3>> if 4 is not working, use reflection to align th. 
   and second, 0.3 in 16 steps)
4. 



SAXSyo = SAXSy.position
WAXSyo = WAXSy.position
for sample in hol1.getSamples():
 
    sample.gotoOrigin()
    while smx.moving==True:
        time.sleep(1)
    sample.measure(10, extra='pos1')
WAXSy.move(WAXSyo+5.16)
SAXSy.move(SAXSyo+5.16)     

for sample in hol1.getSamples():

    sample.gotoOrigin()
    while smx.moving==True:
        time.sleep(1)
    sample.measure(10, extra='pos2')
WAXSy.move(WAXSyo)
SAXSy.move(SAXSyo)     

'''
