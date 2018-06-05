import pandas as pd

def preproc_fuel1(_df):
    df = pd.DataFrame(_df)
    cols = [
        'Cog.Cog', 'Sog.Sog', 'Hdg.TrueHdg', 'Stw.Stw', 'ROT.Value', 'Depth.Depth', 'WindTrue.Angle',
        'WindTrue.Speed', 'TSDNmeaProcessor.Sway_a', 'TSDNmeaProcessor.Surge_a',
        'TSDNmeaProcessor.Heave_a', 'TSDNmeaProcessor.Roll_w', 'TSDNmeaProcessor.Pitch_w', 'TSDNmeaProcessor.Yaw_w',
        'TSDNmeaProcessor.Roll_d', 'TSDNmeaProcessor.Pitch_d', 'NMEA_CUSTOM1_Port_Shaft_RPM.Value', 'NMEA_CUSTOM1_Port_Pitch.Value',
        'NMEA_CUSTOM1_Stbd_Shaft_RPM.Value', 'NMEA_CUSTOM1_Stbd_Pitch.Value', 'NMEA_CUSTOM1_Thruster1_Pitch.Value', 'NMEA_CUSTOM1_Thruster2_Pitch.Value',
        'NMEA_CUSTOM1_Thruster3_Pitch.Value', 'NMEA_CUSTOM2_RUDDER1.Value', 'NMEA_CUSTOM2_RUDDER2.Value', 'NMEA_CUSTOM2_Thruster1_Force.Value',
        'NMEA_CUSTOM2_Thruster2_Force.Value', 'NMEA_CUSTOM2_Thruster3_Force.Value'
        ]
    df = df.dropna(subset=cols + ['NMEA_CUSTOM3_True_Cons.Value'])
    return (df[cols],
            df['NMEA_CUSTOM3_True_Cons.Value'])


def preproc_fuel2t(_df):
    df = pd.DataFrame(_df)
    cols = [
        'Cog.Cog', 'Sog.Sog', 'Stw.Stw', 'ROT.Value', 'WindTrue.Angle',
        'TSDNmeaProcessor.Roll_w',  'NMEA_CUSTOM1_Port_Shaft_RPM.Value', 'NMEA_CUSTOM1_Port_Pitch.Value',
        'NMEA_CUSTOM1_Stbd_Shaft_RPM.Value', 'NMEA_CUSTOM2_Thruster3_Force.Value'
        ]
    df = df.dropna(subset=cols + ['NMEA_CUSTOM3_True_Cons.Value'])
    return (df[cols],
            df['NMEA_CUSTOM3_True_Cons.Value'])

def preproc_fuel2o(_df):
    df = pd.DataFrame(_df)
    cols = [
        'Sog.Sog', 'Stw.Stw', 'ROT.Value',
        'TSDNmeaProcessor.Pitch_d', 'NMEA_CUSTOM1_Port_Shaft_RPM.Value', 'NMEA_CUSTOM1_Port_Pitch.Value',
        'NMEA_CUSTOM1_Stbd_Shaft_RPM.Value', 'NMEA_CUSTOM1_Stbd_Pitch.Value', 'NMEA_CUSTOM1_Thruster1_Pitch.Value',
        'NMEA_CUSTOM2_Thruster3_Force.Value'
        ]
    df = df.dropna(subset=cols + ['NMEA_CUSTOM3_True_Cons.Value'])
    return (df[cols],
            df['NMEA_CUSTOM3_True_Cons.Value'])