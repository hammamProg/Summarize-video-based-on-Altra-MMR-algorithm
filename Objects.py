import statistics
from enum import Enum


class ExtendedEnum(Enum):
    def __str__(self):
            return str(self.value)
    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))
    
    @classmethod
    def list_name(cls):
        return list(map(lambda c: c.name, cls))
    
class Tour20Vidoes(ExtendedEnum):
    MP = "Machu Picchu" #- 7 videos
    TM = "Taj Mahal" #- 7 videos
    BF = "Basilica of the Sagrada Familia" #- 6 videos
    SB = "St. Peter�s Basilica" #- 5 videos
    MC = "Milan Cathedral" #- 10 videos
    AT = "Alcatraz" #- 6 videos
    GB = "Golden Gate Bridge" #- 6 videos
    ET = "Eiffel Tower" #- 8 videos
    NC = "Notre Dame Cathedral" #- 8 videos
    TA = "The Alhambra" #- 6 videos
    HM = "Hagia Sophia Museum" #- 6 videos
    CB = "Charles Bridge" #- 6 videos
    GM = "Great Wall at Mutianyu" #- 5 videos
    BK = "Burj Khalifa" #- 9 videos
    WP = "Wat Pho" #- 5 videos
    CI = "Chicen Itza" #- 8 videos
    SH = "Sydney Opera House" #- 10 videos
    PT = "Petronas Twin Towers" #- 9 videos
    PC = "Panama Canal" #- 6 videos
    AW = "Angkor Wat" #- 6 videos


class FeaturesType(ExtendedEnum):
    ''' For feature: CNN, HSV'''
    CNN = 'CNN'
    HSV = 'HSV'
    I3D = 'I3D'

class AverageSimilarityMeasure(ExtendedEnum):
    ''' For single frame: AM, GM'''
    AM = 'arithmetic_sum'
    GM = 'geometric_sum'

class ShotValueRepresentation(ExtendedEnum):
    ''' For shot: average, mean, median, stdev, sum'''
    AVERAGE = 'average'
    MEDIAN = 'median'
    STDEV = 'stdev'
    SUM = 'sum'

class Frame:
    def __init__(self, num_video, num_shot, frame_index, features, frame, video_mmr_value_AM=None,video_mmr_value_GM=None):
        self.num_video = num_video
        self.num_shot = num_shot
        self.frame_index = frame_index
        self.features = features
        self.frame = frame
        self.video_mmr_value_AM = video_mmr_value_AM
        self.video_mmr_value_GM = video_mmr_value_GM

class Shot:
    def __init__(self, num_video, num_shot, frames:list[Frame], is_included_in_final_summary:bool=None, i3d_features=None,mmr_sum=None):
        self.num_video = num_video
        self.num_shot = num_shot
        self.frames = frames
        self.is_included_in_final_summary = is_included_in_final_summary
        self.i3d_features = i3d_features
        self.mmr_sum = mmr_sum
   
    def mmr_sum_value(self):  # Give me accuracy => 45.45 , labda = 0.7
        shot_mmr_sum_score = 0
        for frame in self.frames:
            shot_mmr_sum_score+= frame.video_mmr_value_AM
        return shot_mmr_sum_score
    
    def mmr_average_value(self): # Give me accuracy => 31.82 , lambda = 0.7
        num_frames = len(self.frames)
        mmr_sum_score = self.mmr_sum_value()
        return (mmr_sum_score/num_frames)
    
    def mmr_mean_value(self):
        shot_values = [frame.video_mmr_value_AM for frame in self.frames]
        shot_mean = statistics.mean(shot_values)
        return shot_mean
    
    def mmr_median_value(self):
        shot_values = [frame.video_mmr_value_AM for frame in self.frames]
        shot_median = statistics.median(shot_values)
        return shot_median
    
    def mmr_stdev(self):
        shot_values = [frame.video_mmr_value_AM for frame in self.frames]
        shot_stdev = statistics.stdev(shot_values)
        return shot_stdev
    
    # add fthe upove functions using video_mmr_value_GM

    def mmr_sum_value_GM(self):  
        shot_mmr_sum_score = 0
        for frame in self.frames:
            shot_mmr_sum_score+= frame.video_mmr_value_GM
        return shot_mmr_sum_score
    def mmr_average_value_GM(self):
        num_frames = len(self.frames)
        mmr_sum_score = self.mmr_sum_value_GM()
        return (mmr_sum_score/num_frames)
    def mmr_mean_value_GM(self):
        shot_values = [frame.video_mmr_value_GM for frame in self.frames]
        shot_mean = statistics.mean(shot_values)
        return shot_mean
    def mmr_median_value_GM(self):
        shot_values = [frame.video_mmr_value_GM for frame in self.frames]
        shot_median = statistics.median(shot_values)
        return shot_median
    def mmr_stdev_GM(self):
        shot_values = [frame.video_mmr_value_GM for frame in self.frames]
        shot_stdev = statistics.stdev(shot_values)
        return shot_stdev
    
class Video:
    def __init__(self, num_video, title, shots:list[Shot]):
        self.num_video = num_video
        self.title = title
        self.shots:list[Shot] = shots

