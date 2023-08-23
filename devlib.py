
import wave
from attrs import define
from os import path
import attrs
from jsonbourne.pydantic import JsonBaseModel


@attrs.define
class WaveMeta:
    nchannels: int
    sampwidth: int = attrs.field()
    framerate: int
    nframes: int
    comptype: str = 'NONE'
    compname: str = 'not compressed'


class WavParams(JsonBaseModel):
    nchannels: int
    sampwidth: int
    framerate: int
    nframes: int
    comptype: str = 'NONE'
    compname: str = 'not compressed'


class WaveFileInfo(JsonBaseModel):
    params: WavParams
    fspath: str
    size: int

    @classmethod
    def from_fspath(cls, fspath):
        with wave.open(fspath, 'rb') as f:
            params = f.getparams()
            d = params._asdict()

        return cls(
            params=WavParams(**d),
            fspath=fspath,
            size=path.getsize(fspath)
        )


def dev():
    obj = WaveFileInfo.from_fspath('uno.wav')
    print(
        obj
    )

    with wave.open('uno.wav', 'rb') as f:
        print(f)
        params = f.getparams()
        print(params)
        print(params._asdict())
        d = params._asdict()
        print(
            WavParams(
                **d
            )
        )
        print(WaveMeta(**d))


if __name__ == '__main__':
    dev()
