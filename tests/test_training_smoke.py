# -*- coding: utf-8 -*-
import unittest
from pathlib import Path

from kraken.configs import (VGSLRecognitionTrainingConfig,
                            VGSLRecognitionTrainingDataConfig,
                            BLLASegmentationTrainingConfig,
                            BLLASegmentationTrainingDataConfig,
                            ROTrainingConfig,
                            ROTrainingDataConfig)
from kraken.train import (VGSLRecognitionModel,
                          VGSLRecognitionDataModule,
                          BLLASegmentationModel,
                          BLLASegmentationDataModule)
from kraken.lib.ro import ROModel, RODataModule
from kraken.train.utils import KrakenTrainer

RESOURCES = Path(__file__).resolve().parent / 'resources'


class TestTrainingSmoke(unittest.TestCase):
    def _trainer(self):
        return KrakenTrainer(
            accelerator='cpu',
            devices=1,
            max_epochs=1,
            min_epochs=1,
            enable_progress_bar=False,
            enable_summary=False,
            enable_checkpointing=False,
            logger=False,
            num_sanity_val_steps=0,
            fast_dev_run=1,
        )

    def test_recognition_training_class_smoke(self):
        xml = RESOURCES / '170025120000003,0074-lite.xml'
        data_config = VGSLRecognitionTrainingDataConfig(
            training_data=[str(xml)],
            evaluation_data=[str(xml)],
            format_type='xml',
            num_workers=0,
        )
        model_config = VGSLRecognitionTrainingConfig(
            spec='[1,12,0,1 Cr3,3,8 S1(1x0)1,3]',
            batch_size=1,
            quit='fixed',
            epochs=1,
            min_epochs=1,
        )
        dm = VGSLRecognitionDataModule(data_config)
        model = VGSLRecognitionModel(model_config)
        trainer = self._trainer()
        trainer.fit(model, dm)

    def test_segmentation_training_class_smoke(self):
        xml = RESOURCES / '170025120000003,0074-lite.xml'
        data_config = BLLASegmentationTrainingDataConfig(
            training_data=[str(xml)],
            evaluation_data=[str(xml)],
            format_type='xml',
            num_workers=0,
        )
        model_config = BLLASegmentationTrainingConfig(
            spec='[1,240,0,1 Cr3,3,8]',
            quit='fixed',
            epochs=1,
            min_epochs=1,
        )
        dm = BLLASegmentationDataModule(data_config)
        model = BLLASegmentationModel(model_config)
        trainer = self._trainer()
        trainer.fit(model, dm)

    def test_reading_order_training_class_smoke(self):
        xml = RESOURCES / 'page' / 'explicit_ro.xml'
        data_config = ROTrainingDataConfig(
            training_data=[str(xml)],
            evaluation_data=[str(xml)],
            format_type='xml',
            num_workers=0,
            batch_size=4,
            level='baselines',
        )
        model_config = ROTrainingConfig(
            quit='fixed',
            epochs=1,
            min_epochs=1,
        )
        dm = RODataModule(data_config)
        model = ROModel(model_config)
        trainer = self._trainer()
        trainer.fit(model, dm)
