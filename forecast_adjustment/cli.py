"""
Command Line Interface for forecast adjustment system.
"""

import argparse
import logging
import os
from datetime import datetime

from forecast_adjustment.agent import ForecastAgent
from forecast_adjustment.environment import ForecastEnvironment
from forecast_adjustment.trainer import ForecastTrainer
from forecast_adjustment.utils.data_generator import generate_complete_dataset
from forecast_adjustment.data.loader import load_data


def setup_logging(log_file=None, verbose=False):
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    
    handlers = [logging.StreamHandler()]
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    
    return logging.getLogger("ForecastAdjustment")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Forecast Adjustment System")
    
    # Mode selection
    parser.add_argument(
        "--mode", 
        choices=["train", "evaluate", "adjust", "generate-data"], 
        required=True,
        help="Operation mode"
    )
    
    # Data options
    parser.add_argument(
        "--forecast-file", 
        help="Path to forecast data CSV"
    )
    parser.add_argument(
        "--historical-file", 
        help="Path to historical data CSV"
    )
    parser.add_argument(
        "--holiday-file", 
        help="Path to holiday data CSV"
    )
    parser.add_argument(
        "--promotion-file", 
        help="Path to promotion data CSV"
    )
    
    # Data generation options
    parser.add_argument(
        "--num-skus", 
        type=int, 
        default=50,
        help="Number of SKUs to generate (data generation mode)"
    )
    parser.add_argument(
        "--forecast-days", 
        type=int, 
        default=14,
        help="Number of forecast days"
    )
    parser.add_argument(
        "--history-days", 
        type=int, 
        default=60,
        help="Number of historical days (data generation mode)"
    )
    
    # Training options
    parser.add_argument(
        "--episodes", 
        type=int, 
        default=100,
        help="Number of training episodes"
    )
    parser.add_argument(
        "--learning-rate", 
        type=float, 
        default=0.01,
        help="Learning rate for the agent"
    )
    parser.add_argument(
        "--optimize-for", 
        choices=["mape", "bias", "both"], 
        default="both",
        help="Metric to optimize for"
    )
    
    # Output options
    parser.add_argument(
        "--output-dir", 
        default="output",
        help="Output directory"
    )
    parser.add_argument(
        "--model-path", 
        help="Path to save/load model"
    )
    
    # Misc options
    parser.add_argument(
        "--start-date", 
        help="Start date for forecast (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose output"
    )
    
    return parser.parse_args()


def load_data_from_args(args, logger):
    """Load data files specified in command line arguments."""
    forecast_data = None
    historical_data = None
    holiday_data = None
    promotion_data = None
    
    # Check if data generation is requested
    if args.mode == "generate-data":
        logger.info("Generating sample data")
        
        start_date = datetime.now()
        if args.start_date:
            try:
                start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
            except ValueError:
                logger.warning(f"Invalid start date format: {args.start_date}, using current date")
        
        datasets = generate_complete_dataset(
            num_skus=args.num_skus,
            forecast_days=args.forecast_days,
            history_days=args.history_days,
            start_date=start_date,
            output_dir=args.output_dir,
            logger=logger
        )
        
        forecast_data = datasets['forecast']
        historical_data = datasets['historical']
        holiday_data = datasets['holiday']
        promotion_data = datasets['promotion']
        
        return forecast_data, historical_data, holiday_data, promotion_data
    
    # Otherwise, load data from specified files
    if args.forecast_file:
        logger.info(f"Loading forecast data from {args.forecast_file}")
        forecast_data = load_data(args.forecast_file, data_type="forecast")
    else:
        logger.error("Forecast file is required for all modes except generate-data")
        return None, None, None, None
    
    if args.historical_file:
        logger.info(f"Loading historical data from {args.historical_file}")
        historical_data = load_data(args.historical_file, data_type="historical")
    
    if args.holiday_file:
        logger.info(f"Loading holiday data from {args.holiday_file}")
        holiday_data = load_data(args.holiday_file, data_type="holiday")
    
    if args.promotion_file:
        logger.info(f"Loading promotion data from {args.promotion_file}")
        promotion_data = load_data(args.promotion_file, data_type="promotion")
    
    return forecast_data, historical_data, holiday_data, promotion_data


def train_mode(args, forecast_data, historical_data, holiday_data, promotion_data, logger):
    """Run the system in training mode."""
    logger.info("Running in training mode")
    
    # Determine start date
    start_date = None
    if args.start_date:
        start_date = args.start_date
    
    # Create environment
    logger.info("Creating forecast environment")
    env = ForecastEnvironment(
        forecast_data=forecast_data,
        historical_data=historical_data,
        holiday_data=holiday_data,
        promotion_data=promotion_data,
        forecast_horizon=args.forecast_days,
        optimize_for=args.optimize_for,
        start_date=start_date,
        logger=logger
    )
    
    # Get feature dimensions
    feature_dims = env.get_feature_dims()
    total_feature_dim = feature_dims[-1]
    
    # Create agent
    logger.info("Creating forecast agent")
    agent = ForecastAgent(
        feature_dim=total_feature_dim,
        learning_rate=args.learning_rate,
        logger=logger
    )
    
    # Create trainer
    logger.info("Creating forecast trainer")
    trainer = ForecastTrainer(
        agent=agent,
        environment=env,
        output_dir=args.output_dir,
        num_episodes=args.episodes,
        optimize_for=args.optimize_for,
        logger=logger
    )
    
    # Train the agent
    logger.info("Training forecast agent")
    trainer.train(verbose=args.verbose)
    
    # Save the model
    model_path = args.model_path or os.path.join(args.output_dir, "models", "final_model.pkl")
    agent.save(model_path)
    logger.info(f"Model saved to {model_path}")


def evaluate_mode(args, forecast_data, historical_data, holiday_data, promotion_data, logger):
    """Run the system in evaluation mode."""
    logger.info("Running in evaluation mode")
    
    if not args.model_path:
        logger.error("Model path is required for evaluation mode")
        return
    
    # Determine start date
    start_date = None
    if args.start_date:
        start_date = args.start_date
    
    # Create environment
    logger.info("Creating forecast environment")
    env = ForecastEnvironment(
        forecast_data=forecast_data,
        historical_data=historical_data,
        holiday_data=holiday_data,
        promotion_data=promotion_data,
        forecast_horizon=args.forecast_days,
        optimize_for=args.optimize_for,
        start_date=start_date,
        logger=logger
    )
    
    # Load agent
    logger.info(f"Loading agent from {args.model_path}")
    agent = ForecastAgent.load(args.model_path, logger=logger)
    
    # Create trainer
    logger.info("Creating forecast trainer")
    trainer = ForecastTrainer(
        agent=agent,
        environment=env,
        output_dir=args.output_dir,
        optimize_for=args.optimize_for,
        logger=logger
    )
    
    # Evaluate the agent
    logger.info("Evaluating forecast agent")
    trainer.evaluate(num_episodes=10, verbose=args.verbose)


def adjust_mode(args, forecast_data, historical_data, holiday_data, promotion_data, logger):
    """Run the system in forecast adjustment mode."""
    logger.info("Running in adjustment mode")
    
    if not args.model_path:
        logger.error("Model path is required for adjustment mode")
        return
    
    # Determine start date
    start_date = None
    if args.start_date:
        start_date = args.start_date
    
    # Create environment
    logger.info("Creating forecast environment")
    env = ForecastEnvironment(
        forecast_data=forecast_data,
        historical_data=historical_data,
        holiday_data=holiday_data,
        promotion_data=promotion_data,
        forecast_horizon=args.forecast_days,
        optimize_for=args.optimize_for,
        start_date=start_date,
        logger=logger
    )
    
    # Load agent
    logger.info(f"Loading agent from {args.model_path}")
    agent = ForecastAgent.load(args.model_path, logger=logger)
    
    # Create trainer
    logger.info("Creating forecast trainer")
    trainer = ForecastTrainer(
        agent=agent,
        environment=env,
        output_dir=args.output_dir,
        optimize_for=args.optimize_for,
        logger=logger
    )
    
    # Generate adjusted forecasts
    logger.info("Generating adjusted forecasts")
    adjustments = trainer.generate_adjusted_forecasts(num_days=args.forecast_days)
    
    # Save adjustments
    adjustments_path = os.path.join(args.output_dir, "adjusted_forecasts.csv")
    adjustments.to_csv(adjustments_path, index=False)
    
    logger.info(f"Adjusted forecasts saved to {adjustments_path}")


def main():
    """Main entry point for the CLI."""
    # Parse command line arguments
    args = parse_args()
    
    # Set up logging
    log_file = os.path.join(args.output_dir, "forecast_adjustment.log")
    logger = setup_logging(log_file, args.verbose)
    
    logger.info("Forecast Adjustment System")
    logger.info(f"Mode: {args.mode}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data or generate data
    forecast_data, historical_data, holiday_data, promotion_data = load_data_from_args(args, logger)
    
    # Check if data is available
    if args.mode != "generate-data" and forecast_data is None:
        logger.error("No forecast data available. Exiting.")
        return 1
    
    # Execute selected mode
    if args.mode == "train":
        train_mode(args, forecast_data, historical_data, holiday_data, promotion_data, logger)
    elif args.mode == "evaluate":
        evaluate_mode(args, forecast_data, historical_data, holiday_data, promotion_data, logger)
    elif args.mode == "adjust":
        adjust_mode(args, forecast_data, historical_data, holiday_data, promotion_data, logger)
    elif args.mode == "generate-data":
        logger.info("Data generation completed")
    
    logger.info("Forecast Adjustment System completed successfully")
    return 0


if __name__ == "__main__":
    exit(main())