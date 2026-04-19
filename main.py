import logging

from dataset_builder.infrastructure.dependency_injection import DependenciesContainer


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    container = DependenciesContainer()
    reader = container.reader()
    ids = reader.list_ids()
    processor = container.processor()
    results = processor.process(ids)

    total_segments = 0
    total_duration = 0.0
    total_files = 0
    issues = []

    for lesson in results:
        if lesson.skip_reason:
            issues.append((lesson.id, lesson.skip_reason))
            continue
        if lesson.segment_result:
            sr = lesson.segment_result
            total_segments += len(sr.segments)
            total_duration += sum(seg.end - seg.start for seg in sr.segments)
            if sr.mismatch_info:
                issues.append((lesson.id, sr.mismatch_info))
        if lesson.dataset:
            total_files += 1

    hours, remainder = divmod(int(total_duration), 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"\n=== Dataset Summary ===")
    print(f"  files    : {total_files}")
    print(f"  segments : {total_segments}")
    print(f"  duration : {hours:02d}:{minutes:02d}:{seconds:02d}")
    if issues:
        print(f"  issues ({len(issues)}):")
        for lesson_id, info in issues:
            print(f"    [{lesson_id}] {info}")
    else:
        print(f"  issues   : none")


if __name__ == "__main__":
    main()
