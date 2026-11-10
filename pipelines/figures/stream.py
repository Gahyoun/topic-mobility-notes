"""Turn annual aggregate stream projections into a standalone browser view."""
import argparse
import json
from pathlib import Path

import numpy as np

DIMENSIONS=(32,64,128,256,512,1024)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('tensor_directory',type=Path)
    p.add_argument('output',type=Path)
    a=p.parse_args()
    data={}
    for d in DIMENSIONS:
        path=a.tensor_directory/f'd{d:04}'/'stream_display.json'
        fields=json.loads(path.read_text())
        vector=np.array([v[:4] for years in fields.values() for dots in years.values() for v in dots])
        points=np.concatenate((vector[:,:2],vector[:,:2]+2*vector[:,2:]),axis=0)
        lo,hi=points.min(0),points.max(0)
        width=hi-lo
        bounds=[lo[0]-.04*width[0],hi[0]+.04*width[0],lo[1]-.04*width[1],hi[1]+.04*width[1]]
        data[d]={'fields':fields,'bounds':bounds}
    template=Path(__file__).with_suffix('.html').read_text()
    a.output.write_text(template.replace('@@STREAM@@',json.dumps(data,separators=(',',':'))))


if __name__=='__main__':
    main()
