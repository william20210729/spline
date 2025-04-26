import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


# This is a bulk method because other methods are being called in this method and nothing more.
def data_preprocessing(input_file_path,columns):
  dataframe = pd.read_csv(input_file_path)
  print(dataframe)
  data = data_normalization2(dataframe,columns)
  print(type(data))
  data.to_csv(output_file,index=False  )

# The dataset values are being normalized in the range(0,1). This will make the model to
# learn fast and easily.

def data_normalization2(arr, columns):
  print(arr.columns)
  scaler = MinMaxScaler()
  #scaler.fit(arr)
  #arr = scaler.transform(arr)
  arr[columns] = scaler.fit_transform(arr[columns])
  return arr

input_file_path = 'ETTh1forNbeatsx.csv'
output_file= 'normalized/ETTh1forNbeatsx.csv'
columns=['OT', 'HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL']
data_preprocessing(input_file_path,columns)


input_file_path = 'ETTh2forNbeatsx.csv'
output_file= 'normalized/ETTh2forNbeatsx.csv'
columns=['OT', 'HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL']
data_preprocessing(input_file_path,columns)

input_file_path = 'ETTm1forNbeatsx.csv'
output_file= 'normalized/ETTm1forNbeatsx.csv'
columns=['OT', 'HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL']
data_preprocessing(input_file_path,columns)

input_file_path = 'ETTm2forNbeatsx.csv'
output_file= 'normalized/ETTm2forNbeatsx.csv'
columns=['OT', 'HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL']
data_preprocessing(input_file_path,columns)


input_file_path = 'WTHforNbeatsx.csv'
output_file= 'normalized/WTHforNbeatsx.csv'
columns=[
         "WetBulbCelsius","Visibility","DryBulbFarenheit","DryBulbCelsius",
         "WetBulbFarenheit","DewPointFarenheit","DewPointCelsius","RelativeHumidity",
         "WindSpeed","WindDirection","StationPressure","Altimeter"         ]
data_preprocessing(input_file_path,columns)


input_file_path = 'ECLforNbeatsx.csv'
output_file= 'normalized/ECLforNbeatsx.csv'
columns=[
    "MT_320","MT_000","MT_001","MT_002","MT_003","MT_004","MT_005",
    "MT_006","MT_007","MT_008","MT_009","MT_010","MT_011","MT_012",
    "MT_013","MT_014","MT_015","MT_016","MT_017","MT_018","MT_019",
    "MT_020","MT_021","MT_022","MT_023","MT_024","MT_025","MT_026",
    "MT_027","MT_028","MT_029","MT_030","MT_031","MT_032","MT_033",
    "MT_034","MT_035","MT_036","MT_037","MT_038","MT_039","MT_040",
    "MT_041","MT_042","MT_043","MT_044","MT_045","MT_046","MT_047",
    "MT_048","MT_049","MT_050","MT_051","MT_052","MT_053","MT_054",
    "MT_055","MT_056","MT_057","MT_058","MT_059","MT_060","MT_061",
    "MT_062","MT_063","MT_064","MT_065","MT_066","MT_067","MT_068",
    "MT_069","MT_070","MT_071","MT_072","MT_073","MT_074","MT_075",
    "MT_076","MT_077","MT_078","MT_079","MT_080","MT_081","MT_082",
    "MT_083","MT_084","MT_085","MT_086","MT_087","MT_088","MT_089",
    "MT_090","MT_091","MT_092","MT_093","MT_094","MT_095","MT_096",
    "MT_097","MT_098","MT_099","MT_100","MT_101","MT_102","MT_103",
    "MT_104","MT_105","MT_106","MT_107","MT_108","MT_109","MT_110",
    "MT_111","MT_112","MT_113","MT_114","MT_115","MT_116","MT_117",
    "MT_118","MT_119","MT_120","MT_121","MT_122","MT_123","MT_124",
    "MT_125","MT_126","MT_127","MT_128","MT_129","MT_130","MT_131",
    "MT_132","MT_133","MT_134","MT_135","MT_136","MT_137","MT_138",
    "MT_139","MT_140","MT_141","MT_142","MT_143","MT_144","MT_145",
    "MT_146","MT_147","MT_148","MT_149","MT_150","MT_151","MT_152",
    "MT_153","MT_154","MT_155","MT_156","MT_157","MT_158","MT_159",
    "MT_160","MT_161","MT_162","MT_163","MT_164","MT_165","MT_166",
    "MT_167","MT_168","MT_169","MT_170","MT_171","MT_172","MT_173",
    "MT_174","MT_175","MT_176","MT_177","MT_178","MT_179","MT_180",
    "MT_181","MT_182","MT_183","MT_184","MT_185","MT_186","MT_187",
    "MT_188","MT_189","MT_190","MT_191","MT_192","MT_193","MT_194",
    "MT_195","MT_196","MT_197","MT_198","MT_199","MT_200","MT_201",
    "MT_202","MT_203","MT_204","MT_205","MT_206","MT_207","MT_208",
    "MT_209","MT_210","MT_211","MT_212","MT_213","MT_214","MT_215",
    "MT_216","MT_217","MT_218","MT_219","MT_220","MT_221","MT_222",
    "MT_223","MT_224","MT_225","MT_226","MT_227","MT_228","MT_229",
    "MT_230","MT_231","MT_232","MT_233","MT_234","MT_235","MT_236",
    "MT_237","MT_238","MT_239","MT_240","MT_241","MT_242","MT_243",
    "MT_244","MT_245","MT_246","MT_247","MT_248","MT_249","MT_250",
    "MT_251","MT_252","MT_253","MT_254","MT_255","MT_256","MT_257",
    "MT_258","MT_259","MT_260","MT_261","MT_262","MT_263","MT_264",
    "MT_265","MT_266","MT_267","MT_268","MT_269","MT_270","MT_271",
    "MT_272","MT_273","MT_274","MT_275","MT_276","MT_277","MT_278",
    "MT_279","MT_280","MT_281","MT_282","MT_283","MT_284","MT_285",
    "MT_286","MT_287","MT_288","MT_289","MT_290","MT_291","MT_292",
    "MT_293","MT_294","MT_295","MT_296","MT_297","MT_298","MT_299",
    "MT_300","MT_301","MT_302","MT_303","MT_304","MT_305","MT_306",
    "MT_307","MT_308","MT_309","MT_310","MT_311","MT_312","MT_313",
    "MT_314","MT_315","MT_316","MT_317","MT_318","MT_319"         ]
data_preprocessing(input_file_path,columns)

