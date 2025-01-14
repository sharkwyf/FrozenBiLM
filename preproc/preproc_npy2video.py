import os
import bz2
import cv2
import numpy as np
import glob
import math
from io import BytesIO
from tqdm import tqdm

if __name__ == '__main__':
    path = "/mnt/nfs/wangyuanfu/data/*.nbz"
    out_path = "/mnt/nfs/wangyuanfu/data/"
    lens = []
    for name in tqdm(glob.glob(path)):
        print(name)
        # load text
        txt_path = name[:-4] + ".txt"
        if os.path.exists(txt_path):
            with open(txt_path) as f:
                text = f.read()
                splits = [item.split("|") for item in text.split("\n")]
                err = [x for x in splits[:-1] if len(x) != 3]
                if len(err) > 0:
                    print(err)
                splits = [x for x in splits if len(x) == 3]
                captions = {
                    "start": np.array([float(x[0])for x in splits], dtype=np.float16),
                    "end": np.array([float(x[1])for x in splits], dtype=np.float16),
                    "word": np.array([x[2]for x in splits]),
                }
                masked = (captions["start"] >= -3) & (captions["start"] <= 3)
                lens.append(sum(masked))
                text = " ".join(captions["word"][masked])
                if lens[-1] > 30:
                    print(lens[-1], text)
        else:
            continue

        # load video
        with open(name, "rb") as f:
            stream = bz2.decompress(f.read())
            frames = np.load(BytesIO(stream))

        result = cv2.VideoWriter(f"{name[:-4]}.avi",
            cv2.VideoWriter_fourcc(*'MJPG'),
            4, frames.shape[1:3][::-1])
        font = cv2.FONT_HERSHEY_SIMPLEX
        for frame in frames:
            interval = 28
            for i in range(math.ceil(len(text) / interval)):
                cv2.putText(frame, 
                    "{}".format(text[i * interval : (i + 1) * interval]), 
                    (10, 15 + 20 * i), 
                    font, 0.5, 
                    (0, 255, 255), 
                    1, 
                    cv2.LINE_4)
            result.write(frame[..., ::-1])
                
        # release the cap object
        result.release()
        # close all windows
        cv2.destroyAllWindows()

        

        print("Succeed")