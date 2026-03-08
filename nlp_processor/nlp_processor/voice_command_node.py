import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import speech_recognition as sr
from jamo import h2j, j2hcj

class VoiceCommandNode(Node):
    def __init__(self):
        super().__init__('voice_command_node')
        self.publisher_ = self.create_publisher(String, '/target_hanguls', 10)
        self.recognizer = sr.Recognizer()
        self.get_logger().info('음성인식 스타트')

        self.timer = self.create_timer(1.0, self.listen_and_publish)

    def listen_and_publish(self):
        with sr.Microphone() as source:
            try:
                # 소음 제거하고 2초 동안 대기
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio = self.recognizer.listen(source, timeout=2.0, phrase_time_limit=3.0)

                # 구글 SST 실행
                text = self.recognizer.recognize_google(audio, language='ko-KR')
                self.get_logger().info(f'인식 결과: {text}')

                # 자음이랑 모음 분리
                jamo_str = j2hcj(h2j(text))
                jamo_list = list(jamo_str)

                msg = String()
                msg.data = ','.join(jamo_list)
                self.publisher_.publish(msg)
                self.get_logger().info(f'분리된 자모: {msg.data}')

            except sr.UnknownValueError:
                pass # 말 안 하면 무시
            except Exception as e:
                self.get_logger().error(f'에러: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = VoiceCommandNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()