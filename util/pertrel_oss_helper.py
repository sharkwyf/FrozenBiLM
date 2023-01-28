from petrel_client.client import Client
from multiprocessing import Queue


class PetrelClient(Client):
    """A wrapper of petrel_client.Client with objects operations"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ENDPOINT_URL = 'http://10.140.14.204:80'
    def traverse_dir(self, url: str):
        return list(self.list(url))

    def check_file(self, url: str):
        return self.contains(url)
        
    def save_npy(self, url: str, file):
        import numpy
        from io import BytesIO
        with BytesIO() as f:
            file = numpy.asanyarray(file)
            numpy.lib.format.write_array(f, file, allow_pickle=True, pickle_kwargs=dict(fix_imports=True))
            self.put(url, f.getvalue())

    def save_video(self, local_file: str, remote_url: str, remote_dir: str):
        import os 
        os.system("aws s3 mb {} --endpoint-url={}".format(remote_dir, self.ENDPOINT_URL))
        os.system("aws s3 mv {} {} --endpoint-url={}".format(local_file, remote_url, self.ENDPOINT_URL))

    def save_npy_direct(self, url: str, nparray):
        import numpy
        from io import BytesIO
        with BytesIO() as f:
            # file = numpy.asanyarray(file)
            numpy.lib.format.write_array(f, nparray, allow_pickle=True, pickle_kwargs=dict(fix_imports=True))
            self.put(url, f.getvalue())

    def save_npz(self, url: str, content):
        import numpy
        from io import BytesIO
        with BytesIO() as f:
            content = numpy.asanyarray(content)
            numpy.savez_compressed(f, content=content)
            self.put(url, f.getvalue())
            
    def save_txt(self, url: str, file):
        from io import BytesIO
        with BytesIO() as f:
            f.write(file.encode('utf-8'))
            self.put(url, f.getvalue())
    
    def save_image(self, local_dir:str, remote_url:str):
        import os 
        # os.system("aws s3 mb {} --endpoint-url={}".format(remote_dir, self.ENDPOINT_URL))
        os.system("aws s3 cp {} {} --recursive --endpoint-url={}".format(local_dir, remote_url, self.ENDPOINT_URL))
    
    def load_nbz(self, url: str):
        import bz2
        import numpy
        from io import BytesIO
        stream = bz2.decompress(self.get(url))
        return numpy.load(BytesIO(stream), allow_pickle=True)
        
    def save_nbz(self, url: str, content):
        import bz2
        import numpy
        from io import BytesIO
        with BytesIO() as stream:
            numpy.save(stream, content)
            compressed = bz2.compress(stream.getvalue())
        self.put(url, compressed)

    def load_npz(self, url: str):
        import numpy
        from io import BytesIO
        stream = BytesIO(self.get(url))   # enable_stream=True for large data
        return numpy.load(stream, allow_pickle=True)["content"]
        
    def load_txt(self, url: str, decoding: str = "utf-8") -> str:
        import numpy
        from io import BytesIO
        stream = BytesIO(self.get(url))   # enable_stream=True for large data
        return stream.read().decode(decoding)


    def load_video(self, url: str):
        import cv2
        presigned_url = self.generate_presigned_url(url, client_method ='get_object', expires_in=3000)
        return cv2.VideoCapture(presigned_url)

    def load_vtt(self, url: str, decoding: str = "utf-8"):
        import webvtt
        from io import StringIO
        payload = self.get(url)
        try:
            payload = payload.decode(decoding)
        except Exception as e:
            return
        raw_vtt = webvtt.read_buffer(StringIO(payload))
        tem = []
        for caption in raw_vtt.captions:
            caption.text = caption.text.split('\n')[-1]
            if len(caption.text) > 1:
                tem.append(caption)
        raw_vtt._captions = tem
        return raw_vtt

    def load_npy(self, url: str, encoding: str = "bytes"):
        import numpy
        from io import BytesIO
        stream = BytesIO(self.get(url))   # enable_stream=True for large data
        return numpy.load(stream, encoding=encoding)
            
    def load_json(self, url: str, decoding: str = "utf-8"):
        import json 
        _raw_json_list = self.get(url).decode(decoding).split('\n')[:-1]
        length_json = len(_raw_json_list)
        # print(_raw_json_list)
        action_list = []
        for i in range(length_json):
            tmp = json.loads(_raw_json_list[i])
            action_list.append(tmp)
        return action_list


class ClientArray(list):
    def __init__(self, arr: list):
        super.__init__(arr)
        self._indices = 1

    def get():
        return
        
            
def init_clients(n_process: int):
    return [PetrelClient() for _ in range(n_process)]
