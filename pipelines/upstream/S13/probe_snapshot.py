import io, json, random, re, subprocess, time
from pathlib import Path
import pyarrow.parquet as pq

class RangeReader(io.RawIOBase):
    def __init__(self, url, length):
        self.url=url
        self.length=length
        self.pos=0
        self.transferred=0
    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos
    def seek(self,offset,whence=0):
        base=(0,self.pos,self.length)[whence]
        self.pos=base+offset
        return self.pos
    def read(self,size=-1):
        end=min(self.pos+size if size>=0 else self.length,self.length)
        if end<=self.pos: return b''
        # Each range gets a distinct URL to avoid intermediaries caching a different byte range.
        start=self.pos
        p=subprocess.run(['/usr/bin/curl','--fail','--silent','--show-error','--max-time','40',
                          '--range',f'{start}-{end-1}',self.url+f'?range={start}-{end-1}'],
                         check=True,capture_output=True)
        if len(p.stdout)!=end-start: raise ValueError(f'Invalid HTTP range: expected {end-start}, got {len(p.stdout)}')
        self.pos=end
        self.transferred+=len(p.stdout)
        return p.stdout
if __name__=='__main__':
    manifest=json.loads(Path('oa-parquet-manifest.json').read_bytes())
    subset=random.Random(6731).sample(manifest['files'],5)
    for sample in subset:
        url=sample['url'].replace('s3://openalex/', 'https://openalex.s3.amazonaws.com/')
        reader=RangeReader(url,sample['meta']['content_length'])
        start=time.monotonic()
        f=pq.ParquetFile(reader, pre_buffer=False)
        print('rows:',f.metadata.num_rows,'rowgroups:',f.metadata.num_row_groups,'size:',reader.length,
              'columns:', [n for n in f.schema_arrow.names if n in ('referenced_works','publication_year','id','is_xpac')],flush=True)
        cols=('id','publication_year','referenced_works','is_xpac')
        chunks=[f.metadata.row_group(g).column(c) for g in range(f.metadata.num_row_groups) for c in range(f.metadata.num_columns)]
        print([(col,sum(x.total_compressed_size for x in chunks if x.path_in_schema.split('.')[0]==col)) for col in cols],flush=True)
        if sample==subset[0]:
            t=f.read_row_group(0,columns=cols,use_threads=False)
            print('types',t.schema,'bytes',reader.transferred,'seconds',time.monotonic()-start,flush=True)
