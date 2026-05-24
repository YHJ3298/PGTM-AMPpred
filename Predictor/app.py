# -*- coding: utf-8 -*-
import sys
from pathlib import Path

import pandas as pd
import torch

from PySide6.QtCore import QObject, QThread, Signal, Slot, Qt
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QFileDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QLineEdit,
    QTextEdit,
    QTableWidget,
    QTableWidgetItem,
    QProgressBar,
    QSpinBox,
    QComboBox,
    QMessageBox,
    QGroupBox,
    QHeaderView,
)

from config import (
    PROJECT_ROOT,
    FIXED_THRESHOLD,
    DEFAULT_BATCH_SIZE,
    DEFAULT_MAX_LENGTH,
    OUTPUT_DIR,
)

from predictor_engine import AMPPredictorEngine


class PredictionWorker(QObject):
    log_signal = Signal(str)
    progress_signal = Signal(int)
    finished_signal = Signal(object)
    error_signal = Signal(str)

    def __init__(
        self,
        fasta_path: str,
        batch_size: int,
        max_length: int,
        device_text: str,
    ):
        super().__init__()

        self.fasta_path = fasta_path
        self.batch_size = batch_size
        self.max_length = max_length
        self.device_text = device_text

    def _resolve_device(self):
        if self.device_text == "CUDA":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA not detected in current environment, cannot use GPU.")
            return torch.device("cuda")

        if self.device_text == "CPU":
            return torch.device("cpu")

        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @Slot()
    def run(self):
        try:
            device = self._resolve_device()

            self.log_signal.emit("=" * 70)
            self.log_signal.emit("PGTM-AMPpred Predictor started.")
            self.log_signal.emit(f"Input FASTA : {self.fasta_path}")
            self.log_signal.emit(f"Device      : {device}")
            self.log_signal.emit(f"Batch size  : {self.batch_size}")
            self.log_signal.emit(f"Max length  : {self.max_length}")
            self.log_signal.emit(f"Threshold   : {FIXED_THRESHOLD}")
            self.log_signal.emit("=" * 70)

            engine = AMPPredictorEngine(device=device)

            def log_callback(msg: str):
                self.log_signal.emit(str(msg))

            def progress_callback(current: int, total: int):
                if total <= 0:
                    self.progress_signal.emit(0)
                else:
                    value = int(current / total * 100)
                    self.progress_signal.emit(value)

            df = engine.predict_fasta(
                fasta_path=self.fasta_path,
                batch_size=self.batch_size,
                max_length=self.max_length,
                threshold=FIXED_THRESHOLD,
                save_csv=True,
                output_csv=str(OUTPUT_DIR / "prediction_result.csv"),
                log_callback=log_callback,
                progress_callback=progress_callback,
            )

            self.progress_signal.emit(100)
            self.finished_signal.emit(df)

        except Exception as e:
            self.error_signal.emit(str(e))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("PGTM-AMPpred Predictor")
        self.resize(1150, 760)

        self.result_df = None
        self.thread = None
        self.worker = None

        self._build_ui()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QVBoxLayout(central)

        title = QLabel("PGTM-AMPpred Predictor")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet(
            "font-size: 26px; font-weight: bold; padding: 10px;"
        )
        main_layout.addWidget(title)

        subtitle = QLabel(
            "A Qt-based predictor integrating ProTrek650M feature extraction and Gated-Adapter enhanced TabM classification"
        )
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet("font-size: 13px; color: #555; padding-bottom: 8px;")
        main_layout.addWidget(subtitle)

        # ======================
        # Input panel
        # ======================
        input_group = QGroupBox("Input FASTA")
        input_layout = QVBoxLayout(input_group)

        file_row = QHBoxLayout()

        self.fasta_line = QLineEdit()
        default_fasta = PROJECT_ROOT / "examples" / "example.fasta"
        if default_fasta.exists():
            self.fasta_line.setText(str(default_fasta))
        else:
            self.fasta_line.setPlaceholderText("Select a FASTA file...")

        self.browse_btn = QPushButton("Browse")
        self.browse_btn.clicked.connect(self.browse_fasta)

        file_row.addWidget(QLabel("FASTA file:"))
        file_row.addWidget(self.fasta_line)
        file_row.addWidget(self.browse_btn)

        input_layout.addLayout(file_row)

        param_row = QHBoxLayout()

        self.batch_spin = QSpinBox()
        self.batch_spin.setMinimum(1)
        self.batch_spin.setMaximum(1024)
        self.batch_spin.setValue(DEFAULT_BATCH_SIZE)

        self.maxlen_spin = QSpinBox()
        self.maxlen_spin.setMinimum(50)
        self.maxlen_spin.setMaximum(10000)
        self.maxlen_spin.setValue(DEFAULT_MAX_LENGTH)

        self.device_combo = QComboBox()
        self.device_combo.addItems(["Auto", "CUDA", "CPU"])

        self.threshold_line = QLineEdit(str(FIXED_THRESHOLD))
        self.threshold_line.setReadOnly(True)

        param_row.addWidget(QLabel("Batch size:"))
        param_row.addWidget(self.batch_spin)

        param_row.addWidget(QLabel("Max length:"))
        param_row.addWidget(self.maxlen_spin)

        param_row.addWidget(QLabel("Device:"))
        param_row.addWidget(self.device_combo)

        param_row.addWidget(QLabel("Threshold:"))
        param_row.addWidget(self.threshold_line)

        input_layout.addLayout(param_row)

        main_layout.addWidget(input_group)

        # ======================
        # Action panel
        # ======================
        action_row = QHBoxLayout()

        self.predict_btn = QPushButton("Run Prediction")
        self.predict_btn.setStyleSheet(
            "font-size: 15px; font-weight: bold; padding: 8px;"
        )
        self.predict_btn.clicked.connect(self.run_prediction)

        self.export_btn = QPushButton("Export CSV")
        self.export_btn.clicked.connect(self.export_csv)
        self.export_btn.setEnabled(False)

        action_row.addWidget(self.predict_btn)
        action_row.addWidget(self.export_btn)

        main_layout.addLayout(action_row)

        # ======================
        # Progress and summary
        # ======================
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        main_layout.addWidget(self.progress_bar)

        self.summary_label = QLabel("Summary: No prediction results yet.")
        self.summary_label.setAlignment(Qt.AlignLeft)
        self.summary_label.setStyleSheet(
            "font-size: 13px; padding: 6px; color: #333;"
        )
        main_layout.addWidget(self.summary_label)

        # ======================
        # Result table
        # ======================
        result_group = QGroupBox("Prediction Results")
        result_layout = QVBoxLayout(result_group)

        self.table = QTableWidget()
        result_layout.addWidget(self.table)

        main_layout.addWidget(result_group, stretch=3)

        # ======================
        # Log panel
        # ======================
        log_group = QGroupBox("Runtime Log")
        log_layout = QVBoxLayout(log_group)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)

        log_layout.addWidget(self.log_text)

        main_layout.addWidget(log_group, stretch=2)

    def browse_fasta(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select FASTA File",
            str(PROJECT_ROOT / "examples"),
            "FASTA Files (*.fasta *.fa *.faa *.txt);;All Files (*)",
        )

        if file_path:
            self.fasta_line.setText(file_path)

    def append_log(self, msg: str):
        self.log_text.append(msg)

    def run_prediction(self):
        fasta_path = self.fasta_line.text().strip()

        if not fasta_path:
            QMessageBox.warning(self, "Warning", "Please select a FASTA file first.")
            return

        if not Path(fasta_path).exists():
            QMessageBox.warning(self, "Warning", "FASTA file does not exist.")
            return

        self.predict_btn.setEnabled(False)
        self.export_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.table.clear()
        self.table.setRowCount(0)
        self.table.setColumnCount(0)
        self.log_text.clear()
        self.summary_label.setText("Summary: Prediction is running...")

        self.thread = QThread()

        self.worker = PredictionWorker(
            fasta_path=fasta_path,
            batch_size=self.batch_spin.value(),
            max_length=self.maxlen_spin.value(),
            device_text=self.device_combo.currentText(),
        )

        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)

        self.worker.log_signal.connect(self.append_log)
        self.worker.progress_signal.connect(self.progress_bar.setValue)
        self.worker.finished_signal.connect(self.on_prediction_finished)
        self.worker.error_signal.connect(self.on_prediction_error)

        self.worker.finished_signal.connect(self.thread.quit)
        self.worker.error_signal.connect(self.thread.quit)

        self.thread.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        self.thread.start()

    def on_prediction_finished(self, df):
        self.result_df = df
        self.display_results(df)
        self.update_summary(df)

        self.predict_btn.setEnabled(True)
        self.export_btn.setEnabled(True)

        self.append_log("Prediction completed successfully.")
        self.append_log(f"Default result saved to: {OUTPUT_DIR / 'prediction_result.csv'}")

        QMessageBox.information(
            self,
            "Done",
            "Prediction completed successfully. Results are displayed in the table.",
        )

    def on_prediction_error(self, error_msg: str):
        self.predict_btn.setEnabled(True)
        self.export_btn.setEnabled(False)
        self.progress_bar.setValue(0)

        self.summary_label.setText("Summary: Prediction failed.")
        self.append_log(f"ERROR: {error_msg}")

        QMessageBox.critical(
            self,
            "Error",
            error_msg,
        )

    def display_results(self, df: pd.DataFrame):
        show_cols = [
            "protein_name",
            "sequence",
            "sequence_length",
            "prob",
            "pred",
            "prediction_label",
        ]

        header_labels = [
            "Protein Name",
            "Original Sequence",
            "Length",
            "Probability",
            "Pred",
            "Label",
        ]

        self.table.setColumnCount(len(show_cols))
        self.table.setRowCount(len(df))
        self.table.setHorizontalHeaderLabels(header_labels)

        for row_idx, (_, row) in enumerate(df.iterrows()):
            for col_idx, col in enumerate(show_cols):
                value = row[col]

                if col == "prob":
                    value = f"{float(value):.6f}"

                item = QTableWidgetItem(str(value))
                item.setTextAlignment(Qt.AlignCenter)

                # Original sequence may be very long - show full sequence on hover
                if col == "sequence":
                    item.setToolTip(str(value))
                    item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)

                self.table.setItem(row_idx, col_idx, item)

        # Make table headers resize appropriately
        header = self.table.horizontalHeader()
        header.setStretchLastSection(False)

        # Protein name - resize to content
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)

        # Sequence - take main horizontal space
        header.setSectionResizeMode(1, QHeaderView.Stretch)

        # Remaining columns - resize to content
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(5, QHeaderView.ResizeToContents)

        self.table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.setWordWrap(False)

    def update_summary(self, df: pd.DataFrame):
        total = len(df)
        amp_count = int((df["pred"] == 1).sum())
        non_amp_count = int((df["pred"] == 0).sum())

        if total > 0:
            amp_ratio = amp_count / total * 100
            non_amp_ratio = non_amp_count / total * 100
        else:
            amp_ratio = 0
            non_amp_ratio = 0

        self.summary_label.setText(
            f"Summary: Total = {total} | "
            f"AMP = {amp_count} ({amp_ratio:.2f}%) | "
            f"non-AMP = {non_amp_count} ({non_amp_ratio:.2f}%) | "
            f"Threshold = {FIXED_THRESHOLD}"
        )

    def export_csv(self):
        if self.result_df is None or self.result_df.empty:
            QMessageBox.warning(
                self,
                "Warning",
                "No prediction results available for export.",
            )
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Prediction Result",
            str(OUTPUT_DIR / "prediction_result.csv"),
            "CSV Files (*.csv);;All Files (*)",
        )

        if file_path:
            self.result_df.to_csv(
                file_path,
                index=False,
                encoding="utf-8-sig",
            )

            QMessageBox.information(
                self,
                "Saved",
                f"Results saved to:\n{file_path}",
            )


def main():
    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()