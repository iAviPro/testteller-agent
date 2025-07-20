"""CLI commands for TestTeller automation generation."""

import asyncio
import logging
from pathlib import Path
from typing import Optional

import typer
from typing_extensions import Annotated

# Import core utilities
from ..core.utils.loader import with_progress_bar_sync
from ..core.data_ingestion.unified_document_parser import UnifiedDocumentParser, DocumentType
from ..core.constants import SUPPORTED_LANGUAGES, SUPPORTED_FRAMEWORKS

# Import automation components  
from .parser.markdown_parser import MarkdownTestCaseParser
from .generators import PythonTestGenerator, JavaScriptTestGenerator, JavaTestGenerator, TypeScriptTestGenerator

logger = logging.getLogger(__name__)


def get_generator(language: str, framework: str, output_dir: Path):
    """Get the appropriate generator based on language and framework."""
    if language == 'python':
        return PythonTestGenerator(framework, output_dir)
    elif language == 'javascript':
        return JavaScriptTestGenerator(framework, output_dir)
    elif language == 'typescript':
        return TypeScriptTestGenerator(framework, output_dir)
    elif language == 'java':
        return JavaTestGenerator(framework, output_dir)
    else:
        raise ValueError(f"Unsupported language: {language}")


def validate_framework(language: str, framework: str) -> bool:
    """Validate that the framework is supported for the language."""
    return framework in SUPPORTED_FRAMEWORKS.get(language, [])


def automate_command(
    input_file: Annotated[str, typer.Argument(help="Path to test cases file (supports .md, .txt, .pdf, .docx, .xlsx)")],
    language: Annotated[str, typer.Option(
        "--language", "-l", help="Programming language for test automation")] = None,
    framework: Annotated[str, typer.Option(
        "--framework", "-F", help="Test framework to use")] = None,
    output_dir: Annotated[str, typer.Option(
        "--output-dir", "-o", help="Output directory for generated tests")] = "./generated_tests",
    interactive: Annotated[bool, typer.Option(
        "--interactive", "-i", help="Interactive mode to select test cases")] = False,
    enhance: Annotated[bool, typer.Option(
        "--enhance", "-E", help="Use LLM to enhance generated test code")] = False,
    llm_provider: Annotated[str, typer.Option(
        "--llm-provider", "-p", help="LLM provider for enhancement (gemini, openai, claude, llama)")] = None
):
    """Generate automation code from TestTeller test cases."""
    
    # Validate input file exists
    input_path = Path(input_file)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_file}")
        print(f"❌ Error: Test cases file '{input_file}' not found.")
        raise typer.Exit(code=1)
    
    # Use defaults from environment or fallback to sensible defaults
    if not language:
        # Try to get from environment
        import os
        language = os.getenv('AUTOMATION_LANGUAGE', 'python')  # Default to python
        print(f"💡 Using default language: {language}")
        print("   (Use --language to specify a different language)")
    
    if not framework:
        # Try to get from environment or use first supported framework for the language
        import os
        framework = os.getenv('AUTOMATION_FRAMEWORK')
        if not framework:
            frameworks = SUPPORTED_FRAMEWORKS.get(language, ['pytest'])
            framework = frameworks[0]  # Use first available framework
        print(f"💡 Using default framework: {framework}")
        print("   (Use --framework to specify a different framework)")
    
    # Validate framework
    if not validate_framework(language, framework):
        print(f"❌ Error: Framework '{framework}' is not supported for language '{language}'.")
        print(f"Supported frameworks for {language}: {', '.join(SUPPORTED_FRAMEWORKS[language])}")
        print("\n💡 To configure defaults, run: testteller configure")
        raise typer.Exit(code=1)
    
    # Show configuration summary
    print(f"\n✅ Configuration: {language}/{framework} → {output_dir}")
    
    # LLM Enhancement is enabled by default (no prompting)
    if not enhance:
        try:
            from .llm_enhancer import is_llm_enhancement_available
            
            if is_llm_enhancement_available():
                enhance = True  # Always enable if available
                if not llm_provider:
                    llm_provider = None  # Use default
        except (ImportError, Exception):
            pass  # LLM enhancement not available, continue without it
    
    try:
        # Parse test cases using unified parser
        file_extension = input_path.suffix.lower()
        
        if file_extension not in ['.md', '.txt', '.pdf', '.docx', '.xlsx']:
            print(f"❌ Unsupported file format: {file_extension}")
            print("Supported formats: .md, .txt, .pdf, .docx, .xlsx")
            raise typer.Exit(code=1)
        
        # Use unified parser for all formats
        unified_parser = UnifiedDocumentParser()
        
        # Parse document for automation with progress
        def parse_operation():
            return asyncio.run(unified_parser.parse_for_automation(input_path))
        
        parsed_doc = with_progress_bar_sync(
            parse_operation, 
            f"📖 Parsing {input_file} ({file_extension})..."
        )
        
        # Extract test cases
        test_cases = parsed_doc.test_cases
        
        # If no structured test cases found, try to use content for context
        if not test_cases:
            if parsed_doc.metadata.document_type in [DocumentType.TEST_CASES, DocumentType.REQUIREMENTS]:
                # For markdown files, try the legacy parser as fallback
                if file_extension == '.md':
                    def fallback_parse():
                        md_parser = MarkdownTestCaseParser()
                        return md_parser.parse_file(input_path)
                    
                    test_cases = with_progress_bar_sync(
                        fallback_parse,
                        "🔄 Trying markdown-specific parser..."
                    )
                
                if not test_cases:
                    print("❌ No test cases found in the file.")
                    print("💡 Tip: Ensure the file contains structured test cases in the expected format.")
                    print("   For supported formats, see: https://github.com/testteller/docs")
                    raise typer.Exit(code=1)
            else:
                print("❌ This document doesn't appear to contain test cases.")
                print(f"   Detected type: {parsed_doc.metadata.document_type.value}")
                print("💡 Try using a file that contains structured test cases.")
                raise typer.Exit(code=1)
        
        # Show parsing results after completion
        print(f"\n✅ Parsing complete!")
        print(f"   • Found {len(test_cases)} test cases")
        if parsed_doc.metadata.title:
            print(f"   • Document: {parsed_doc.metadata.title}")
        if parsed_doc.metadata.sections:
            print(f"   • Sections: {len(parsed_doc.metadata.sections)}")
        print(f"   • Content: {parsed_doc.metadata.word_count} words, {parsed_doc.metadata.character_count} characters")
        
        # Interactive selection if requested
        if interactive:
            selected_cases = interactive_select_tests(test_cases)
            if not selected_cases:
                print("❌ No test cases selected.")
                raise typer.Exit(code=1)
            test_cases = selected_cases
        
        # Generate automation code
        output_path = Path(output_dir)
        generator = get_generator(language, framework, output_path)
        
        # Generate with progress bar
        def generate_operation():
            return generator.generate(test_cases)
        
        generated_files = with_progress_bar_sync(
            generate_operation,
            f"🚀 Generating {language}/{framework} tests..."
        )
        
        # LLM Enhancement (optional)
        if enhance:
            try:
                from .llm_enhancer import create_enhancer, is_llm_enhancement_available
                
                if not is_llm_enhancement_available():
                    print("⚠️  LLM enhancement not available. Ensure LLM is configured in TestTeller.")
                    print("   Generated code will be used without enhancement.")
                else:
                    enhancer = create_enhancer(provider=llm_provider)
                    if enhancer.is_available():
                        # Enhance with progress bar
                        def enhance_operation():
                            return enhancer.enhance_generated_tests(
                                generated_files, language, framework
                            )
                        
                        enhanced_files = with_progress_bar_sync(
                            enhance_operation,
                            f"🤖 Enhancing code with {enhancer.provider or 'AI'}..."
                        )
                        
                        # Count enhanced files
                        enhanced_count = sum(1 for filename, content in enhanced_files.items() 
                                           if content != generated_files.get(filename, ""))
                        
                        if enhanced_count > 0:
                            generated_files = enhanced_files
                            print(f"✅ Enhanced {enhanced_count} test files using {enhancer.provider}")
                            
                            # Show optimization suggestions
                            suggestions = enhancer.get_optimization_suggestions(language, framework)
                            if suggestions:
                                print(f"\n💡 Optimization suggestions for {language} + {framework}:")
                                for i, suggestion in enumerate(suggestions[:5], 1):  # Show top 5
                                    print(f"  {i}. {suggestion}")
                        else:
                            print("⚠️  No files were enhanced. Using original generated code.")
                    else:
                        print("⚠️  LLM enhancer initialization failed. Using original generated code.")
                        
            except ImportError:
                print("⚠️  LLM enhancement dependencies not available. Using original generated code.")
            except Exception as e:
                logger.warning(f"LLM enhancement failed: {e}")
                print(f"⚠️  LLM enhancement failed: {e}")
                print("   Using original generated code.")
        
        # Write files with progress bar
        def write_operation():
            return generator.write_files(generated_files)
        
        with_progress_bar_sync(
            write_operation,
            f"📝 Writing {len(generated_files)} files to {output_dir}..."
        )
        
        # Summary
        print(f"\n🎉 Automation Complete!")
        print(f"✅ Successfully generated {len(generated_files)} files:")
        for file_name in generated_files:
            print(f"   • {output_path / file_name}")
        
        # Next steps
        print_next_steps(language, framework, output_path)
        
    except Exception as e:
        logger.error(f"Failed to generate automation code: {e}", exc_info=True)
        print(f"\n❌ Error: {e}")
        raise typer.Exit(code=1)


def interactive_select_tests(test_cases):
    """Interactive test case selection."""
    print("\n📋 Available test cases:")
    for i, tc in enumerate(test_cases, 1):
        print(f"{i:3d}. [{tc.id}] {tc.objective[:60]}...")
    
    print("\nSelect test cases to automate:")
    print("  • Enter numbers separated by commas (e.g., 1,3,5)")
    print("  • Enter ranges (e.g., 1-5)")
    print("  • Enter 'all' to select all tests")
    print("  • Enter 'none' to cancel")
    
    selection = typer.prompt("\nYour selection").strip().lower()
    
    if selection == 'none':
        return []
    elif selection == 'all':
        return test_cases
    else:
        selected_indices = parse_selection(selection, len(test_cases))
        return [test_cases[i-1] for i in selected_indices]


def parse_selection(selection: str, max_index: int) -> list:
    """Parse user selection string into list of indices."""
    indices = set()
    
    for part in selection.split(','):
        part = part.strip()
        if '-' in part:
            # Range
            try:
                start, end = map(int, part.split('-'))
                indices.update(range(max(1, start), min(max_index + 1, end + 1)))
            except ValueError:
                print(f"⚠️  Invalid range: {part}")
        else:
            # Single number
            try:
                num = int(part)
                if 1 <= num <= max_index:
                    indices.add(num)
                else:
                    print(f"⚠️  Index out of range: {num}")
            except ValueError:
                print(f"⚠️  Invalid number: {part}")
    
    return sorted(indices)


def print_next_steps(language: str, framework: str, output_dir: Path):
    """Print next steps based on language and framework."""
    print("\n📚 Next steps:")
    
    if language == 'python':
        print(f"  1. cd {output_dir}")
        print("  2. pip install -r requirements.txt")
        if framework == 'pytest':
            print("  3. pytest")
        else:
            print("  3. python -m unittest discover")
    
    elif language == 'javascript':
        print(f"  1. cd {output_dir}")
        print("  2. npm install")
        print("  3. npm test")
    
    elif language == 'java':
        print(f"  1. cd {output_dir}")
        print("  2. mvn clean install")
        print("  3. mvn test")
    
    print("\n💡 Tips:")
    print("  • Review and customize the generated tests")
    print("  • Update test data and assertions")
    print("  • Configure test environment settings")
    print("  • Add missing test implementations marked with TODO")