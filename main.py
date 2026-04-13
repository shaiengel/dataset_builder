from dotenv import load_dotenv

from dataset_builder.infrastructure.dependency_injection import DependenciesContainer

LESSON_IDS = ["151828"]


def main():
    load_dotenv()
    container = DependenciesContainer()
    processor = container.processor()
    results = processor.process(LESSON_IDS)

    for lesson in results:
        print(f"\n=== Lesson {lesson.id} ===")
        if lesson.transcript:
            print(f"  transcript words: {len(lesson.transcript.words)}")
        if lesson.vtt:
            print(f"  vtt cues        : {len(lesson.vtt.cues)}")
        if lesson.segment_result:
            sr = lesson.segment_result
            print(f"  segments        : {len(sr.segments)}")
            print(f"  aligned words   : {sr.total_words_aligned}")
            print(f"  status          : {sr.status.value}")
            if sr.truncate_at is not None:
                print(f"  truncate at     : {sr.truncate_at:.3f}s")
            if sr.mismatch_info:
                print(f"  mismatch info   : {sr.mismatch_info}")
        if lesson.wav_path:
            print(f"  wav_path        : {lesson.wav_path}")
        if lesson.dataset:
            print(f"  dataset rows    : {len(lesson.dataset)}")


if __name__ == "__main__":
    main()
