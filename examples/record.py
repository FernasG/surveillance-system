import os
import ffmpeg

def record_segmented(segment_time=10):
    try:
        # Entrada de Vídeo
        v_in = ffmpeg.input(
            '/dev/video0', 
            f='v4l2', 
            input_format='mjpeg',
            video_size='1280x720',
            framerate='30'
        )
        
        # Entrada de Áudio
        a_in = ffmpeg.input('plughw:CARD=WEBCAM,DEV=0', f='alsa')

        file_template = os.path.join("videos", "cam1_%s.mp4")

        # Configuração do Output com Segmentação
        out = ffmpeg.output(
            v_in,
            a_in,
            file_template,
            vcodec='libx264',
            preset='veryfast',
            pix_fmt='yuv420p',
            acodec='aac',
            # Aqui entra a mágica do segment:
            f='segment',
            segment_time=segment_time,
            reset_timestamps=1,
            segment_format='mp4',
            strftime=1
        ).overwrite_output()

        print(f"Gravando em: {file_template}") 

        return out.run_async()
    except ffmpeg.Error as e:
        # O ffmpeg-python joga o erro do stderr aqui se algo der errado
        print("Erro no FFmpeg:", e.stderr.decode())
