import json
import time
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import os
from glob import glob




class DataTransformer: 
    def __init__(self,json_directory,common_senses):
        self.json_file_paths = []    
        self.total_files=0
        self.total_time=0
        for common_sense in common_senses:
            self.json_file_paths.extend(glob(os.path.join(json_directory, common_sense, '*.json')))

    # JSON 파일을 로드하여 데이터 추출
    def load_json_files(self):
        json_datas = []
        total_files = len(self.json_file_paths)
        total_time = 0
    # 파일별로 진행 상황을 모니터링하기 위해 tqdm 사용


    # 파일별로 진행 상황을 모니터링하기 위해 tqdm 사용
        for i, json_file_path in enumerate(tqdm(self.json_file_paths, desc="Loading JSON files", unit="file")):
            start_time = time.time()

        with open(json_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            if isinstance(data, list):  # 데이터가 리스트인 경우 확장
                json_datas.extend(data)
            else:  # 데이터가 리스트가 아닌 경우 추가
                json_datas.append(data)

            end_time = time.time()
            elapsed_time = end_time - start_time
            total_time += elapsed_time
            print(f"File {i+1}/{total_files} processed in {elapsed_time:.2f} seconds")

        return json_datas, total_time

#documents = [{"text": json.dumps(item)} for item in json_datas]