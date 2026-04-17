"""
Main application entry point for the Vision-Accelerated Exam Data Entry System.

This module initializes the application, sets up logging, and launches
the appropriate interface (GUI or CLI).
"""

import sys
import argparse
from pathlib import Path
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config import get_config, SystemConfig
from app.utils import setup_logging, get_logger
from app.services import PipelineService, ExportService


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Vision-Accelerated Exam Data Entry System"
    )
    
    parser.add_argument(
        "--mode",
        choices=["gui", "cli"],
        default="gui",
        help="Application mode (default: gui)"
    )
    
    parser.add_argument(
        "--input",
        type=Path,
        help="Input directory or file path"
    )
    
    parser.add_argument(
        "--output",
        type=Path,
        help="Output file path for export"
    )
    
    parser.add_argument(
        "--format",
        choices=["excel", "csv", "json"],
        default="excel",
        help="Export format (default: excel)"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["debug", "info", "warning", "error"],
        default="info",
        help="Logging level (default: info)"
    )
    
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to configuration file"
    )
    
    return parser.parse_args()


def run_cli(args: argparse.Namespace) -> int:
    """
    Run the application in CLI mode.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        Exit code
    """
    logger = get_logger(__name__)
    
    if not args.input:
        logger.error("No input specified. Use --input to specify input path.")
        return 1
    
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input path does not exist: {input_path}")
        return 1
    
    # Initialize pipeline
    pipeline = PipelineService()
    export_service = ExportService()
    
    # Process input
    logger.info(f"Processing: {input_path}")
    
    if input_path.is_file():
        results = [pipeline.process_image(input_path)]
    else:
        results = pipeline.process_directory(input_path)
    
    # Report results
    stats = pipeline.get_stats()
    logger.info(f"Processed: {stats.total_processed}")
    logger.info(f"Successful: {stats.successful}")
    logger.info(f"Failed: {stats.failed}")
    logger.info(f"Success rate: {stats.success_rate:.1f}%")
    
    # Export if output specified
    if args.output and results:
        records = [r.record for r in results if r.success and r.record]
        
        if args.format == "excel":
            success = export_service.to_excel(records, args.output)
        elif args.format == "csv":
            success = export_service.to_csv(records, args.output)
        else:
            success = export_service.to_json(records, args.output)
        
        if success:
            logger.info(f"Exported to: {args.output}")
        else:
            logger.error("Export failed")
            return 1
    
    return 0


def run_gui() -> int:
    """
    Run the application in GUI mode.
    
    Returns:
        Exit code
    """
    from app.ui import run_ui
    return run_ui()


def main() -> int:
    """
    Main application entry point.
    
    Returns:
        Exit code
    """
    args = parse_arguments()
    
    # Setup logging
    log_level = getattr(logging, args.log_level.upper())
    config = get_config()
    log_file = config.directories.log_dir / "vision_system.log"
    setup_logging(log_level=log_level, log_file=log_file)
    
    logger = get_logger(__name__)
    logger.info("=" * 60)
    logger.info("Vision-Accelerated Exam Data Entry System")
    logger.info("=" * 60)
    
    # Load custom config if specified
    if args.config:
        logger.info(f"Loading config from: {args.config}")
        # Config loading would be implemented here
    
    # Run in appropriate mode
    if args.mode == "cli":
        return run_cli(args)
    else:
        return run_gui()


if __name__ == "__main__":
    sys.exit(main())