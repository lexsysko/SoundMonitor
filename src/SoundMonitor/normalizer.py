import logging
import numpy as np

from SoundMonitor.enums import NormalizeMethod

logger = logging.getLogger(__name__)


def get_normalize(a: np.ndarray, method: NormalizeMethod | None = None) -> np.float64 | None:
    match method:
        case None | NormalizeMethod.NONE:
            return None

        case NormalizeMethod.MAX:
            return a.max().astype(np.float64)

        case NormalizeMethod.LINALG | NormalizeMethod.LOG_LINALG:
            return np.linalg.norm(a)
    return None


def get_concatenated_norm(
    a: list[np.ndarray] | np.ndarray, b: list[np.ndarray] | np.ndarray, method: NormalizeMethod
) -> np.float64 | None:
    # concatenate both lists into one array
    all_psd = np.concatenate([np.ravel(a), np.ravel(b)])
    return get_normalize(all_psd, method)


def normalize_psd(
    psd: np.ndarray | list[np.ndarray], method: NormalizeMethod | None = None, normalize_value=None
) -> tuple[np.ndarray, bool, np.float64 | None]:

    if isinstance(psd, list):
        logger.debug(f"normalize_psd concatenated")
        psd = np.concatenate(psd)

    match method:
        case None | NormalizeMethod.NONE:
            return psd.astype(np.float32), True, None

        case NormalizeMethod.MAX | NormalizeMethod.LINALG:
            norm_val = normalize_value or get_normalize(psd, method)
            if norm_val is None or norm_val < 1e-16:
                return np.zeros_like(psd), False, None
            return (psd / norm_val).astype(np.float32), True, norm_val

        case NormalizeMethod.LOG_LINALG:
            psd_log = 10 * np.log10(psd + 1e-16)
            logger.debug(f"{psd_log.max()=}- {psd_log.min()=} = {psd_log.max() - psd_log.min()}")
            if psd_log.max() - psd_log.min() > 3:
                norm_val = normalize_value or get_normalize(psd_log, method)
                if norm_val > 0:
                    return (psd_log / norm_val).astype(np.float32), True, norm_val
            return np.zeros_like(psd_log), False, None

    # Fallback for unknown method
    return np.zeros_like(psd), False, None
