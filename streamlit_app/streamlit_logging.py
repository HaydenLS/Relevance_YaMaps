import logging
import time
from collections import deque
from typing import Deque, Optional

import streamlit as st

class StreamlitLogHandler(logging.Handler):
    """
    Пишет логи в st.empty()
    Хранит последние N строк в session_state
    """
    def __init__(self, 
        placeholder,
        key: str = "live_logs",
        max_lines: int = 500,
        level: int = logging.INFO
    ):

        super().__init__(level=level)
        self.placeholder = placeholder
        self.key = key
        self.max_lines = max_lines

        if self.key not in st.session_state:
            st.session_state[self.key] = deque(maxlen=self.max_lines)

        # Формат сообщения
        fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        self.setFormatter(logging.Formatter(fmt))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            line = self.format(record)

            logs: Deque[str] = st.session_state[self.key]
            logs.append(line)

            # Обновляем UI
            self.placeholder.code("\n".join(logs), language="text")

            # иногда полезно дать Streamlit шанс прорендерить
            time.sleep(0.001)
        except Exception:
            # не даём логированию ронять приложение
            pass


def setup_logger_for_streamlit(
    placeholder,
    logger_name: str = "org_relevance",
    level: int = logging.INFO
) -> logging.Logger:
    
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)

    # Важно: не плодить хендлеры при каждом rerun
    logger.handlers = []
    logger.propagate = False

    handler = StreamlitLogHandler(placeholder, level=level)
    logger.addHandler(handler)
    return logger