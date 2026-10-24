class ContextBuilder:
    def build(
        self,
        results: list[dict],
    ) -> str:
        parts = []

        for result in results:
            timestamp = result["timestamp"]
            text = result["text"]

            if isinstance(timestamp, list):
                time_label = (
                    f"{timestamp[0]:.1f}-{timestamp[1]:.1f}s"
                )
            else:
                time_label = f"{timestamp:.1f}s"

            part = f"[{time_label}]"

            if text:
                part += f"\nTranscript: {text}"

            parts.append(part)

        return "\n\n".join(parts)